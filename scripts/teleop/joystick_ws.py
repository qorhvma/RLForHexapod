"""우노 조이스틱 → 라즈베리파이 노드(HexaCore robot_main.py) 조종. WebSocket + JSON.

실물 로봇의 파이 노드(`HexaCore-clone/software/raspberrypi/robot_main.py`)는 `ws://<파이 IP>:8765` 로
`{"cmd":"walk","vx":..,"vy":..,"wz":..}` 를 받아 삼각(tripod) 보행으로 걷는다 (0.5초 안에 다시 안 오면 자동 정지).
이 프로그램은 노트북(또는 아무 PC)에 꽂은 우노 조이스틱을 읽어 그 JSON을 20 Hz로 보낸다.
HXCMD v1(UDP, 강화학습 정책용)은 joystick_sender.py 가 맡는다. 두 프로그램은 조이스틱 읽기·매핑을 공유한다.

필요: pip install pyserial websockets

    python3 joystick_ws.py --joytest                          # 로봇 없이 값·버튼 번호만 확인
    python3 joystick_ws.py --host 192.168.0.42                # 파이 IP
    python3 joystick_ws.py --host localhost                   # 파이 노드를 `robot_main.py --sim` 으로 PC에서 돌릴 때

조작 (joystick_sender.py 와 같은 버튼, --btn_* 로 바꾼다)
  조이스틱        기울인 방향으로 걷기 (최고 속도는 파이 노드의 한계 vx 0.12, vy 0.08 m/s, wz 0.6 rad/s)
  Left / Right   왼쪽 / 오른쪽 회전
  조이스틱 짧게   일어서기 (`stand`: 앉은 높이에서 몸을 서기 높이까지 들어 올림). 처음 켠 뒤와 비상정지를 푼 뒤에 쓴다
  노랑 / 파랑     몸 높이 올리기 / 내리기 (`height`: 한 번 10mm, 누르고 있으면 초당 40mm)
  버튼 2          비상정지 켜기 / 끄기. 켜면 `estop`(서보 힘 빼기, 로봇이 주저앉는다). 끄면 `stand`로 다시 일어선다
  (길게 누름 리셋은 이 노드에 해당 명령이 없어서 쓰지 않는다)

안전
  - 움직일 때만 `walk`를 보낸다. 0 속도 `walk`도 노드를 서기 모드로 바꿔 서보를 켜 버리기 때문이다.
    멈추면 `stop`을 한 번 보내고, 그 뒤에는 아무것도 안 보내서 노드의 0.5초 데드맨이 지킨다.
  - 조이스틱 신호가 0.5초 없거나 USB가 빠지면 정지(`stop`). 이 프로그램이 죽거나 네트워크가 끊겨도 데드맨이 멈춘다.
  - 비상정지 중에는 `walk`를 절대 보내지 않는다 (노드는 `walk`를 받으면 비상정지를 풀고 서 버린다).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import time

from joystick_sender import (
    Config,
    Mapper,
    Output,
    SerialSource,
    add_joystick_args,
    add_mapper_args,
    config_from_args,
    find_port,
    fmt,
    joytest,
    require_pyserial,
)

WALK_HZ = 20.0
ESTOP_RESEND = 1.0   # 비상정지 중 `estop`을 다시 보내는 간격 [s]
PING_SEC = 2.0
STATE_STALE = 1.5    # 이 시간 동안 텔레메트리가 없으면 "응답 없음" [s]
WS_PORT = 8765
# 파이 노드(robot_main.py)의 자체 한계. 노드가 어차피 이 값으로 자른다.
# 높이는 상대값으로만 쓴다 (Mapper 의 height 변화량 → {"cmd":"height","delta_mm"}), 범위는 노드가 자른다
PI_CONFIG = Config(vx_max=0.25, vy_max=0.15, wz_max=0.6, height=0.0, h_min=-1.0, h_max=1.0)
HEIGHT_SEND_MM = 5.0   # Mapper 높이가 이만큼 바뀌면 보낸다 [mm]


class WsCommander:
    """조이스틱 Output을 파이 노드 명령(dict) 목록으로 바꾼다. 네트워크를 모르는 순수 로직."""

    def __init__(self):
        self.moving = False
        self.estop = False
        self._last_estop_sent = -math.inf
        self._last_h = None     # 마지막으로 보낸 기준 높이 (Mapper 의 height [m])

    def reconnected(self):
        """다시 연결됐다: 노드는 우리 상태를 모른다. 움직이는 중이었다는 기록을 지우고 비상정지는 바로 다시 알린다."""
        self.moving = False
        self._last_estop_sent = -math.inf
        self._last_h = None

    def messages(self, out: Output, now: float) -> list[dict]:
        msgs: list[dict] = []
        if out.estop:
            if not self.estop or now - self._last_estop_sent >= ESTOP_RESEND:
                msgs.append({"cmd": "estop"})
                self._last_estop_sent = now
            self.estop = True
            self.moving = False
            return msgs  # 비상정지 중에는 walk/stand 를 보내지 않는다
        if self.estop:  # 비상정지를 풀었다 → 다시 일어선다
            self.estop = False
            msgs.append({"cmd": "stand"})
            self._last_h = out.height
        elif out.tap:
            msgs.append({"cmd": "stand"})
            self._last_h = out.height          # 짧게 누르면 Mapper 가 높이를 기본값으로 되돌린다 → 기준만 다시
        if self._last_h is None:
            self._last_h = out.height
        dh = (out.height - self._last_h) * 1000.0
        if abs(dh) >= HEIGHT_SEND_MM - 1e-6:
            msgs.append({"cmd": "height", "delta_mm": round(dh, 1)})
            self._last_h = out.height
        moving = abs(out.vx) + abs(out.vy) + abs(out.wz) > 1e-3
        if moving:
            msgs.append({"cmd": "walk", "vx": round(out.vx, 4), "vy": round(out.vy, 4), "wz": round(out.wz, 4)})
        elif self.moving:
            msgs.append({"cmd": "stop"})
        self.moving = moving
        return msgs


class PiLink:
    """파이 노드 WebSocket 연결. 끊기면 알아서 다시 붙는다 (send/poll은 예외를 내지 않는다)."""

    def __init__(self, url: str, token: str | None, connect=None):
        self.url, self.token = url, token
        self._connect = connect or self._default_connect
        self.ws = None
        self.tel: dict = {}
        self.tel_time = -math.inf
        self.rtt_ms: float | None = None
        self.just_connected = False
        self._next_try = 0.0
        self._warned = False

    @staticmethod
    def _default_connect(url: str):
        from websockets.sync.client import connect
        return connect(url, open_timeout=2, ping_interval=5, ping_timeout=5, max_size=2**22)

    def _open(self, now: float):
        try:
            self.ws = self._connect(self.url)
        except Exception as e:  # noqa: BLE001  연결 거부, 시간 초과, 이름 해석 실패 모두
            self.ws = None
            self._next_try = now + 1.0
            if not self._warned:
                print(f"\n[파이] {self.url} 연결 중... ({e})")
                self._warned = True
            return
        if self.token:
            self._send_raw({"cmd": "auth", "token": self.token})
        self.just_connected = True
        print(f"\n[파이] 연결됨 {self.url}")
        self._warned = False

    def _drop(self, why: Exception, now: float):
        print(f"\n[파이] 연결 끊김: {why}")
        self._warned = True
        if self.ws is not None:
            with contextlib.suppress(Exception):
                self.ws.close()
        self.ws = None
        self._next_try = now + 1.0

    def _send_raw(self, msg: dict):
        self.ws.send(json.dumps(msg, separators=(",", ":")))

    def send(self, msg: dict, now: float):
        if self.ws is None:
            return
        try:
            self._send_raw(msg)
        except Exception as e:  # noqa: BLE001
            self._drop(e, now)

    def poll(self, now: float) -> bool:
        """연결을 유지하고 받은 메시지를 모두 처리한다. 방금 (다시) 연결됐으면 True."""
        self.just_connected = False
        if self.ws is None:
            if now >= self._next_try:
                self._open(now)
            return self.just_connected
        while True:
            try:
                raw = self.ws.recv(timeout=0)
            except TimeoutError:
                break
            except Exception as e:  # noqa: BLE001  ConnectionClosed 등
                self._drop(e, now)
                break
            self._handle(raw, now)
        return self.just_connected

    def _handle(self, raw, now: float):
        try:
            msg = json.loads(raw)
        except (TypeError, ValueError):
            return
        typ = msg.get("type")
        if typ == "telemetry":
            self.tel, self.tel_time = msg, now
        elif typ == "hello":
            gait = msg.get("gait")
            print(f"\n[파이] {msg.get('robot')} v{msg.get('version')} sim={msg.get('simulated')} 보행={gait or '?'}"
                  + (f" 코드={msg['code']}" if msg.get("code") else "")
                  + (" (토큰 필요: --token 또는 HEXAPOD_TOKEN)" if msg.get("auth_required") and not self.token else ""))
            if gait != "ik+calib":
                print("[파이] ★ 예전 노드입니다 (캘리브레이션 보정 없음). "
                      "파이에서 hexapod.service 가 옛 코드를 돌리고 있지 않은지 확인할 것")
        elif typ == "pong" and msg.get("t") is not None:
            self.rtt_ms = (time.time() - msg["t"]) * 1000.0
        elif typ == "log":
            print(f"\n[파이] {msg.get('line')}")
        elif typ == "ack" and not msg.get("ok"):
            print(f"\n[파이] 명령 실패 {msg.get('cmd')}: {msg.get('error')}")

    def close(self):
        if self.ws is not None:
            with contextlib.suppress(Exception):
                self.ws.close()
            self.ws = None


def status_line(out: Output, alive: bool, link: PiLink, now: float) -> str:
    line = f"명령 vx {fmt(out.vx)} vy {fmt(out.vy)} wz {fmt(out.wz)}"
    if not alive:
        line += " | 조이스틱 신호 없음 → 정지"
    if link.ws is None:
        line += " | 파이 연결 안 됨"
    elif now - link.tel_time > STATE_STALE:
        line += " | 파이 응답 없음"
    else:
        t = link.tel
        r = t.get("robot", {})
        feet = "".join("●" if f else "○" for f in r.get("foot", []))
        line += f" | [{t.get('mode')}] 발 {feet} R {r.get('roll', 0):+.1f} P {r.get('pitch', 0):+.1f}"
        gs = t.get("gait")
        if gs:
            line += f" 높이 {gs.get('height', 0):.0f}"
            if abs(gs.get("height", 0) - gs.get("height_target", 0)) > 0.5:
                line += f"→{gs.get('height_target', 0):.0f}"
            line += "mm"
        if link.rtt_ms is not None:
            line += f" | {link.rtt_ms:.0f}ms"
        if t.get("safety"):
            line += f" ⚠ {','.join(t['safety'])}"
    if out.estop:
        line += " [비상정지]"
    return line


def main():
    p = argparse.ArgumentParser(description="우노 조이스틱으로 파이 노드(WebSocket) 조종")
    add_joystick_args(p)
    p.add_argument("--host", default="raspberrypi.local", help="파이 IP 또는 이름")
    p.add_argument("--ws_port", type=int, default=WS_PORT)
    p.add_argument("--token", default=os.environ.get("HEXAPOD_TOKEN"), help="파이 노드의 HEXAPOD_TOKEN")
    add_mapper_args(p, PI_CONFIG, with_height=False)
    a = p.parse_args()
    cfg = config_from_args(a, PI_CONFIG)
    require_pyserial()
    port = find_port() if a.port == "auto" else a.port
    src = SerialSource(port)
    print(f"조이스틱 {port}")
    if a.joytest:
        joytest(src, cfg)
        src.close()
        return

    try:
        import websockets  # noqa: F401
        import websockets.sync.client  # noqa: F401
    except ImportError:
        raise SystemExit("websockets 필요 (버전 11 이상): pip install websockets") from None

    mapper, cmdr = Mapper(cfg), WsCommander()
    link = PiLink(f"ws://{a.host}:{a.ws_port}", a.token)
    print(f"파이 {link.url}   (Ctrl+C 종료)")
    t_start = time.monotonic()
    hinted = False
    period = 1.0 / WALK_HZ
    next_time = time.perf_counter()
    last_ping = 0.0
    frame = 0
    out = Output(0.0, 0.0, 0.0, 0.0, 0, False, False)
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
            if link.poll(now):
                cmdr.reconnected()
            out = mapper.tick(now, alive)
            for msg in cmdr.messages(out, now):
                link.send(msg, now)
            if now - last_ping >= PING_SEC:
                last_ping = now
                link.send({"cmd": "ping", "t": time.time()}, now)

            frame += 1
            if frame % 2 == 0:  # 화면은 10 Hz
                print("\r" + status_line(out, alive, link, now).ljust(130), end="", flush=True)

            next_time += period
            wait = next_time - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            else:
                next_time = time.perf_counter()
    except KeyboardInterrupt:
        pass
    finally:
        link.send({"cmd": "stop"}, time.monotonic())  # 끝날 때 한 번 멈춘다. 사라져도 노드의 데드맨이 멈춘다
        link.close()
        src.close()
        print("\n종료")


if __name__ == "__main__":
    main()
