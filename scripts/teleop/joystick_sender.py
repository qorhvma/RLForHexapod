"""옛 우노 조이스틱 조종기: 라즈베리파이에 USB로 꽂고, HXCMD v1 명령을 50 Hz로 보낸다.

조이스틱은 강화학습 이전(legacy/bluetooth_pc)에 쓰던 것과 같다: 우노(오렌지보드) + 조이스틱 쉴드가 USB 시리얼
115200 baud로 `<x,y,버튼,볼륨>\\n` 을 계속 보낸다 (x,y,볼륨 0~1023, 버튼은 비트마스크). 우노 펌웨어는 그대로 쓴다.
표준 라이브러리 + pyserial만 쓴다 (`pip install pyserial`). 같은 폴더의 protocol.py만 있으면 돈다.

    python3 joystick_sender.py --joytest                   # 로봇 없이 값·버튼 번호만 확인
    python3 joystick_sender.py                             # 포트 자동 선택, 같은 파이의 로봇(127.0.0.1)으로
    python3 joystick_sender.py --port /dev/ttyACM0 --host 192.168.0.10

조작 (버튼 번호는 --joytest 로 확인하고, 다르면 --btn_* 로 바꾼다)
  조이스틱        기울인 방향으로 걷기. 많이 기울일수록 빠름 (가속은 부드럽게). 방향이 반대면 --fwd_sign / --side_sign
  Left / Right   왼쪽 / 오른쪽 회전 (조이스틱과 같이 쓰면 걸으면서 돈다)
  노랑 / 파랑     몸 높이 올리기 / 내리기 (한 번 = --h_step, 누르고 있으면 계속)
  조이스틱 누름   짧게: 높이 기본값.  1초 이상 길게: 리셋 (넘어졌을 때)
  SAFE 버튼       비상정지 켜기 / 끄기 (옛 코드의 안전 모드 버튼 자리). 켜진 동안 속도 0
  볼륨           --pot 을 주면 최고 속도 조절

안전
  - 조이스틱 신호가 0.5초 없거나 USB가 빠지면 STOP 명령(속도 0)을 보내고, 다시 꽂으면 자동으로 이어 붙는다.
  - 이 프로그램이 죽으면 받는 쪽이 0.25초 뒤 스스로 멈춘다 (protocol.py의 타임아웃).
  - 명령을 학습 범위로 자르는 건 받는 쪽이다. 여기 기본 최고 속도는 protocol.Limits 의 기본값과 같다.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import time
from dataclasses import dataclass

from protocol import CMD_HZ, PORT, STATUS_CLAMPED, STATUS_FALLEN, STATUS_TIMEOUT, CommandSender, Limits, Mode

BAUD = 115200
JOY_TIMEOUT = 0.5   # 이 시간 동안 조이스틱 신호가 없으면 끊긴 것으로 본다 [s]
REOPEN_SEC = 1.0    # 시리얼이 끊기면 이 간격으로 다시 연다 [s]
STATE_STALE = 1.0   # 이 시간 동안 로봇 STATE가 없으면 "응답 없음" [s]


@dataclass
class Sample:
    x: int
    y: int
    mask: int
    pot: int = 1023


def parse_line(line: bytes) -> Sample | None:
    """`<x,y,버튼[,볼륨]>` 한 줄. 형식이 틀리면 None."""
    line = line.strip()
    if not (line.startswith(b"<") and line.endswith(b">")):
        return None
    try:
        v = [int(t) for t in line[1:-1].split(b",")]
    except ValueError:
        return None
    if len(v) < 3:
        return None
    return Sample(v[0], v[1], v[2], v[3] if len(v) > 3 else 1023)


class LineParser:
    """바이트 조각을 이어 붙여 완성된 줄만 Sample로 꺼낸다. 중간에 잘린 줄·쓰레기는 버린다."""

    def __init__(self):
        self.buf = b""

    def feed(self, data: bytes) -> list[Sample]:
        self.buf += data
        out = []
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            s = parse_line(line)
            if s is not None:
                out.append(s)
        if len(self.buf) > 1024:  # 줄바꿈 없이 쌓이면 쓰레기
            self.buf = b""
        return out


# ───────────────────────── 시리얼 ─────────────────────────
def find_port() -> str:
    """연결된 USB 시리얼 포트가 하나면 그것을 쓴다."""
    from serial.tools import list_ports
    ports = [p for p in list_ports.comports() if "ACM" in p.device or "USB" in p.device]
    if len(ports) == 1:
        return ports[0].device
    if not ports:
        raise SystemExit("USB 시리얼 포트가 없음 → 조이스틱(우노) USB 케이블을 확인하고 --port 로 지정할 것")
    lines = [f"  {p.device}  ({p.description})" for p in ports]
    raise SystemExit("USB 시리얼 포트가 여러 개 → --port 로 조이스틱을 지정할 것 "
                     "(우노를 뺐다 꽂아서 사라지는 쪽이 조이스틱):\n" + "\n".join(lines))


def _open_help(port: str, e: Exception) -> str:
    msg = f"[조이스틱] {port} 를 못 열었음: {e}"
    if "lock" in str(e).lower():
        msg += "\n  다른 프로그램이 이 포트를 쓰는 중 → 켜 둔 joystick_sender/--joytest/시리얼 모니터를 끌 것"
    if getattr(e, "errno", None) == 13:
        msg += "\n  권한 없음 → 'sudo usermod -aG dialout $USER' 한 뒤 로그아웃·로그인 (한 번만)"
    return msg


class SerialSource:
    """우노 조이스틱에서 Sample을 블로킹 없이 읽는다. 끊기면 알아서 다시 연다 (poll은 예외를 내지 않는다)."""

    def __init__(self, port: str, opener=None, baud: int = BAUD):
        if opener is None:
            import serial

            def opener(p):
                # timeout 0: read가 기다리지 않는다. exclusive: 다른 프로그램이 이미 포트를 쓰면 열지 않는다
                # (두 프로그램이 같은 포트를 읽으면 줄이 반씩 갈라져 신호가 끊긴 것처럼 보인다). 리눅스만 지원
                return serial.Serial(p, baud, timeout=0, exclusive=True if os.name == "posix" else None)
        self.port = port
        self._opener = opener
        self.ser = None
        self.parser = LineParser()
        self.last_sample = -math.inf   # 마지막 샘플 시각 (monotonic)
        self._next_open = 0.0
        self._warned = False
        self._open(time.monotonic(), first=True)

    def _open(self, now: float, first: bool = False):
        try:
            self.ser = self._opener(self.port)
        except Exception as e:  # noqa: BLE001  SerialException, OSError 모두
            if first:
                raise SystemExit(_open_help(self.port, e)) from e
            self.ser = None
            self._next_open = now + REOPEN_SEC
            if not self._warned:
                print(f"\n[조이스틱] {self.port} 다시 연결 중... ({e})")
                self._warned = True
            return
        self.parser = LineParser()
        if self._warned or not first:
            print(f"\n[조이스틱] {self.port} 다시 연결됨")
        self._warned = False

    def poll(self, now: float) -> list[Sample]:
        if self.ser is None:
            if now >= self._next_open:
                self._open(now)
            return []
        try:
            n = getattr(self.ser, "in_waiting", 0)
            data = self.ser.read(n if n else 1)
        except Exception as e:  # noqa: BLE001  USB가 빠짐
            print(f"\n[조이스틱] 끊김: {e}")
            self._warned = True
            self.close()
            self._next_open = now + REOPEN_SEC
            return []
        samples = self.parser.feed(data) if data else []
        if samples:
            self.last_sample = now
        return samples

    def alive(self, now: float) -> bool:
        return now - self.last_sample < JOY_TIMEOUT

    def close(self):
        if self.ser is not None:
            with contextlib.suppress(Exception):
                self.ser.close()
            self.ser = None


# ───────────────────────── 조이스틱 → 명령 ─────────────────────────
@dataclass
class Config:
    vx_max: float = Limits().vx[1]       # 조이스틱 끝까지 밀었을 때 [m/s]
    vy_max: float = Limits().vy[1]
    wz_max: float = Limits().wz[1]       # [rad/s]
    height: float = 0.20                 # 기본 몸 높이 [m]
    h_min: float = Limits().height[0]
    h_max: float = Limits().height[1]
    h_step: float = 0.01                 # 버튼 한 번 [m]
    h_rate: float = 0.04                 # 버튼을 누르고 있을 때 [m/s]
    h_hold: float = 0.35                 # 이보다 오래 누르면 '누르고 있음' [s]
    accel_v: float = 0.5                 # 속도 변화 한계 [m/s²]
    accel_w: float = 1.5                 # [rad/s²]
    deadzone: float = 0.12               # 가운데 무시 구간 (0~1)
    expo: float = 0.5                    # 0 = 그대로, 1 = 가운데 근처를 아주 섬세하게
    fwd_sign: int = -1                   # 위로 밀었는데 뒤로 가면 +1
    side_sign: int = +1                  # 왼쪽으로 밀었는데 오른쪽으로 가면 -1
    center: int = 512
    use_pot: bool = False
    pot_min: float = 0.3                 # 볼륨 맨 아래일 때 속도 배율
    long_press: float = 1.0              # 조이스틱 누름이 이보다 길면 리셋 [s]
    # 버튼 번호 = 우노가 보내는 버튼 값의 비트 (옛 hexapod_joy_smooth.py 와 같다)
    btn_up: int = 0
    btn_down: int = 1
    btn_estop: int = 2
    btn_left: int = 4
    btn_right: int = 5
    btn_stick: int = 6


@dataclass
class Output:
    vx: float
    vy: float
    wz: float
    height: float
    mode: int
    estop: bool
    reset: bool   # 이번 tick에 리셋을 한 번 요청한다
    tap: bool = False   # 이번 tick에 조이스틱 누름을 짧게 눌렀다 뗐다 (한 번만 True)


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


class Mapper:
    """샘플(조이스틱 값)을 명령으로 바꾼다. 하드웨어·네트워크를 모르는 순수 로직이라 따로 시험할 수 있다."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.height = _clamp(cfg.height, cfg.h_min, cfg.h_max)
        self.estop = False
        self.v = [0.0, 0.0, 0.0]   # 지금 내보내는 vx vy wz (가속 한계를 거친 값)
        self._last_tick: float | None = None
        self._reset_inputs()

    def _reset_inputs(self):
        self.x = self.y = self.cfg.center
        self.mask = 0
        self.pot = 1023
        self._have = False       # 이번 연결의 첫 샘플을 받았나 (첫 샘플은 버튼 변화의 기준으로만 쓴다)
        self.pressed = 0         # 새로 눌린 버튼 비트 (tick에서 처리하고 지운다)
        self.released = 0
        self.hold_since: float | None = None   # 높이 버튼을 누르기 시작한 시각
        self.stick_since = 0.0
        self.stick_armed = False  # 조이스틱 누름이 눌려 있고 아직 짧은/긴 누름 처리가 안 됐다

    def feed(self, s: Sample):
        """샘플 하나. 눌림/뗌은 tick 사이에 짧게 끝나도 놓치지 않게 여기서 모은다."""
        if self._have:
            self.pressed |= s.mask & ~self.mask
            self.released |= self.mask & ~s.mask
        self._have = True
        self.x, self.y, self.mask, self.pot = s.x, s.y, s.mask, s.pot

    def held(self, bit: int) -> bool:
        return bool(self.mask & (1 << bit))

    def axes(self) -> tuple[float, float]:
        """(앞뒤, 좌우) -1~1. 좌우 + = 왼쪽. 기본 부호는 옛 코드를 실제 조이스틱에 맞춘 값:
        y가 작을수록(위로 밀수록) 앞, x가 클수록 왼쪽."""
        c = self.cfg

        def ax(v: int, sign: int) -> float:
            a = (v - c.center) / 512.0 * sign
            if abs(a) < c.deadzone:
                return 0.0
            a = _clamp((a - math.copysign(c.deadzone, a)) / (1 - c.deadzone), -1.0, 1.0)
            return a * (1 - c.expo) + a * abs(a) * c.expo
        return ax(self.y, c.fwd_sign), ax(self.x, c.side_sign)

    def tick(self, now: float, alive: bool) -> Output:
        c = self.cfg
        dt = 0.0 if self._last_tick is None else _clamp(now - self._last_tick, 0.0, 0.1)
        self._last_tick = now

        if not alive:  # 조이스틱이 끊겼다: 멈추고, 다시 연결되면 버튼 기준을 새로 잡는다
            self._reset_inputs()
            self.v = [0.0, 0.0, 0.0]
            return Output(0.0, 0.0, 0.0, self.height, Mode.STOP, self.estop, False)

        reset = tap = False
        pressed, released = self.pressed, self.released
        self.pressed = self.released = 0

        if pressed & (1 << c.btn_estop):
            self.estop = not self.estop
        if pressed & (1 << c.btn_up):
            self.height += c.h_step
        if pressed & (1 << c.btn_down):
            self.height -= c.h_step
        hdir = int(self.held(c.btn_up)) - int(self.held(c.btn_down))
        if hdir:
            if self.hold_since is None:
                self.hold_since = now
            elif now - self.hold_since > c.h_hold:
                self.height += hdir * c.h_rate * dt
        else:
            self.hold_since = None

        # 조이스틱 누름: 짧게 = 높이 기본값 (뗄 때), 길게 = 리셋 (1초가 되는 순간 한 번)
        if pressed & (1 << c.btn_stick):
            self.stick_armed, self.stick_since = True, now
        if self.stick_armed and self.held(c.btn_stick) and now - self.stick_since >= c.long_press:
            reset, self.stick_armed = True, False
        if released & (1 << c.btn_stick) and self.stick_armed:
            self.height, self.stick_armed, tap = c.height, False, True
        self.height = _clamp(self.height, c.h_min, c.h_max)

        fwd, side = self.axes()
        mag = math.hypot(fwd, side)
        if mag > 1.0:
            fwd, side = fwd / mag, side / mag
        turn = int(self.held(c.btn_left)) - int(self.held(c.btn_right))
        k = c.pot_min + (1.0 - c.pot_min) * _clamp(self.pot / 1023.0, 0.0, 1.0) if c.use_pot else 1.0
        target = (fwd * c.vx_max * k, side * c.vy_max * k, turn * c.wz_max * k)
        if self.estop:
            target = (0.0, 0.0, 0.0)
            self.v = [0.0, 0.0, 0.0]
        lims = (c.accel_v * dt, c.accel_v * dt, c.accel_w * dt)
        self.v = [v + _clamp(t - v, -a, a) for v, t, a in zip(self.v, target, lims, strict=True)]
        return Output(self.v[0], self.v[1], self.v[2], self.height, Mode.POLICY, self.estop, reset, tap)


# ───────────────────────── 실행 ─────────────────────────
def fmt(v: float, sign: bool = True) -> str:
    if math.isnan(v):
        return "  --  "
    return f"{v:+.2f}" if sign else f"{v:.3f}"


def joytest(src: SerialSource, cfg: Config):
    """로봇 없이 조이스틱 값과 눌린 버튼 번호를 보여 준다. 방향·버튼 번호를 맞출 때 쓴다."""
    print("조이스틱 시험 (Ctrl+C 종료). 눌러 보고 아래 값과 --btn_*, --fwd_sign, --side_sign 이 맞는지 확인")
    m = Mapper(cfg)
    last = None
    try:
        while True:
            now = time.monotonic()
            for s in src.poll(now):
                m.feed(s)
                last = s
            if last is None or not src.alive(now):
                line = "신호 없음 (우노 부팅에 2초쯤 걸린다)"
            else:
                fwd, side = m.axes()
                bits = [str(b) for b in range(8) if last.mask & (1 << b)] or ["-"]
                line = (f"x {last.x:4d} y {last.y:4d} 볼륨 {last.pot:4d} | 앞뒤 {fwd:+.2f} 좌우 {side:+.2f}"
                        f" | 눌린 버튼 번호: {','.join(bits)}")
            print("\r" + line.ljust(100), end="", flush=True)
            time.sleep(0.02)
    except KeyboardInterrupt:
        print()


def add_joystick_args(p: argparse.ArgumentParser):
    p.add_argument("--port", default="auto", help="시리얼 포트 (예: /dev/ttyACM0). auto = USB 시리얼이 하나면 그것")
    p.add_argument("--joytest", action="store_true", help="로봇 없이 조이스틱 값과 버튼 번호만 확인")


def add_mapper_args(p: argparse.ArgumentParser, d: Config, with_height: bool = True):
    """Mapper 설정 인자. d = 기본값 (스크립트마다 최고 속도가 다르다)."""
    p.add_argument("--vx_max", type=float, default=d.vx_max, help="앞뒤 최고 속도 [m/s]")
    p.add_argument("--vy_max", type=float, default=d.vy_max, help="좌우 최고 속도 [m/s]")
    p.add_argument("--wz_max", type=float, default=d.wz_max, help="회전 최고 속도 [rad/s]")
    p.add_argument("--accel_v", type=float, default=d.accel_v, help="속도 변화 한계 [m/s²]")
    p.add_argument("--accel_w", type=float, default=d.accel_w, help="회전 속도 변화 한계 [rad/s²]")
    if with_height:
        p.add_argument("--height", type=float, default=d.height, help="기본 몸 높이 [m]")
        p.add_argument("--h_min", type=float, default=d.h_min)
        p.add_argument("--h_max", type=float, default=d.h_max)
        p.add_argument("--h_step", type=float, default=d.h_step, help="높이 버튼 한 번 [m]")
    p.add_argument("--deadzone", type=float, default=d.deadzone)
    p.add_argument("--expo", type=float, default=d.expo)
    p.add_argument("--fwd_sign", type=int, choices=(-1, 1), default=d.fwd_sign, help="위로 밀었는데 뒤로 가면 1")
    p.add_argument("--side_sign", type=int, choices=(-1, 1), default=d.side_sign,
                   help="왼쪽으로 밀었는데 오른쪽으로 가면 -1")
    p.add_argument("--center", type=int, default=d.center, help="조이스틱 가운데 값 (0~1023)")
    p.add_argument("--pot", action="store_true", help="볼륨으로 최고 속도 조절")
    for name in ("up", "down", "estop", "left", "right", "stick"):
        p.add_argument(f"--btn_{name}", type=int, default=getattr(d, f"btn_{name}"), help="버튼 번호 (비트)")


def config_from_args(a: argparse.Namespace, defaults: Config | None = None) -> Config:
    """인자에 없는 값(예: with_height=False 일 때 높이)은 defaults 에서."""
    d = defaults or Config()
    return Config(a.vx_max, a.vy_max, a.wz_max, getattr(a, "height", d.height), getattr(a, "h_min", d.h_min),
                  getattr(a, "h_max", d.h_max), getattr(a, "h_step", d.h_step),
                  accel_v=a.accel_v, accel_w=a.accel_w, deadzone=a.deadzone, expo=a.expo,
                  fwd_sign=a.fwd_sign, side_sign=a.side_sign, center=a.center, use_pot=a.pot,
                  btn_up=a.btn_up, btn_down=a.btn_down, btn_estop=a.btn_estop, btn_left=a.btn_left,
                  btn_right=a.btn_right, btn_stick=a.btn_stick)


def require_pyserial():
    try:
        import serial  # noqa: F401
    except ImportError:
        raise SystemExit("pyserial 필요: pip install pyserial") from None


def main():
    p = argparse.ArgumentParser(description="우노 조이스틱으로 HXCMD v1 명령 보내기")
    add_joystick_args(p)
    p.add_argument("--host", default="127.0.0.1", help="로봇/시뮬레이션 IP (같은 파이에서 정책을 돌리면 기본값)")
    p.add_argument("--udp_port", type=int, default=PORT)
    add_mapper_args(p, Config())
    a = p.parse_args()
    cfg = config_from_args(a)
    require_pyserial()
    port = find_port() if a.port == "auto" else a.port
    src = SerialSource(port)
    print(f"조이스틱 {port}")
    if a.joytest:
        joytest(src, cfg)
        src.close()
        return

    mapper = Mapper(cfg)
    tx = CommandSender(a.host, a.udp_port)
    print(f"HXCMD v1 → {a.host}:{a.udp_port}   (Ctrl+C 종료)")
    last_state_seq = None
    last_state_time = -math.inf
    t_start = time.monotonic()
    hinted = False
    period = 1.0 / CMD_HZ
    next_time = time.perf_counter()
    frame = 0
    out = Output(0.0, 0.0, 0.0, mapper.height, Mode.STOP, False, False)
    try:
        while True:
            now = time.monotonic()
            for s in src.poll(now):
                mapper.feed(s)
            alive = src.alive(now)
            if not hinted and not alive and src.last_sample == -math.inf and now - t_start > 4.0:
                hinted = True
                print("\n[조이스틱] 4초 넘게 신호가 한 줄도 안 옴 → 우노에 joystick_controller.ino 가 올라가 있는지, "
                      "다른 프로그램이 포트를 쓰고 있지 않은지 확인 (--joytest 로 따로 볼 수 있음)")
            out = mapper.tick(now, alive)
            if out.reset:
                tx.request_reset()
                print("\n[리셋 요청]")
            tx.send(out.vx, out.vy, out.wz, out.height, out.mode, out.estop)

            st = tx.poll_state()
            if st is not None and st.seq != last_state_seq:
                last_state_seq, last_state_time = st.seq, time.perf_counter()

            frame += 1
            if frame % 5 == 0:  # 화면은 10 Hz로 갱신
                line = (f"명령 vx {fmt(out.vx)} vy {fmt(out.vy)} wz {fmt(out.wz)} h {fmt(out.height, False)}")
                if not alive:
                    line += " | 조이스틱 신호 없음 → 정지"
                if time.perf_counter() - last_state_time > STATE_STALE:
                    line += " | 로봇 응답 없음"
                else:
                    m = st.meas
                    line += (f" | 측정 vx {fmt(m[0])} vy {fmt(m[1])} wz {fmt(m[2])} h {fmt(m[3], False)}"
                             f" | {tx.rtt_ms}ms")
                    flags = [t for bit, t in ((STATUS_TIMEOUT, "끊김"), (STATUS_CLAMPED, "잘림"),
                                              (STATUS_FALLEN, "넘어짐")) if st.status & bit]
                    if flags:
                        line += " " + ",".join(flags)
                if out.estop:
                    line += " [비상정지]"
                print("\r" + line.ljust(120), end="", flush=True)

            next_time += period
            wait = next_time - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_time = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        # 끝낼 때 정지 명령을 몇 번 보낸다. 사라져도 받는 쪽이 0.25초 뒤 타임아웃으로 멈춘다
        for _ in range(5):
            tx.send(0.0, 0.0, 0.0, mapper.height, Mode.STOP, mapper.estop)
            time.sleep(period)
        tx.close()
        src.close()
        print("\n종료")


if __name__ == "__main__":
    sys.exit(main())
