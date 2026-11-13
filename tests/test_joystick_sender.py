"""우노 조이스틱 조종기 (scripts/teleop/joystick_sender.py) 시험. 하드웨어·torch·Isaac Lab 없이 돈다."""

import math
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "teleop"))
from joystick_sender import Config, LineParser, Mapper, Sample, SerialSource, parse_line  # noqa: E402
from protocol import CommandReceiver, CommandSender, Mode  # noqa: E402

CENTER = Sample(512, 512, 0)


def bit(n):
    return 1 << n


def run(m, samples, t0=0.0, dt=0.02, alive=True):
    """샘플을 하나씩 먹이며 tick을 돌리고 마지막 Output을 돌려준다."""
    out = None
    for i, s in enumerate(samples):
        m.feed(s)
        out = m.tick(t0 + i * dt, alive)
    return out


def settle(m, s, t0=0.0, n=100):
    """가속 한계를 거친 값이 목표에 닿을 때까지."""
    return run(m, [s] * n, t0)


# ── 파싱 ──
def test_parse_line():
    assert parse_line(b"<512,300,5,1023>\r\n") == Sample(512, 300, 5, 1023)
    assert parse_line(b"<1,2,3>") == Sample(1, 2, 3, 1023)  # 볼륨 없는 형식
    for bad in (b"512,300,5,1023", b"<1,2>", b"<a,b,c>", b"", b"<1,2,3"):
        assert parse_line(bad) is None, bad


def test_line_parser_joins_chunks_and_drops_garbage():
    p = LineParser()
    assert p.feed(b"<5,5,0,9") == []  # 아직 줄이 안 끝남
    out = p.feed(b"99>\nxx<0,0,0,0>\n<1,2,3,4>\nfoo\n<7,8")  # 앞에 쓰레기가 붙은 줄·foo는 버림
    assert out == [Sample(5, 5, 0, 999), Sample(1, 2, 3, 4)]
    assert p.feed(b",9,1>\n") == [Sample(7, 8, 9, 1)]
    assert p.feed(b"z" * 2000) == [] and p.buf == b""  # 줄바꿈 없이 쌓이면 비운다


# ── 매핑 ──
def test_center_is_zero():
    out = settle(Mapper(Config()), CENTER)
    assert (out.vx, out.vy, out.wz) == (0.0, 0.0, 0.0) and out.mode == Mode.POLICY and not out.estop


def test_stick_directions_and_scale():
    c = Config()
    out = settle(Mapper(c), Sample(512, 0, 0))  # 위로 끝까지 → 앞
    assert out.vx == pytest.approx(c.vx_max) and abs(out.vy) < 1e-9
    out = settle(Mapper(c), Sample(512, 1023, 0))  # 아래 → 뒤
    assert out.vx < -0.9 * c.vx_max
    # 옛 코드(STICK_SIDE_SIGN=+1)에서 실제 하드웨어로 맞춰 둔 방향: x가 클수록 왼쪽
    out = settle(Mapper(c), Sample(1023, 512, 0))  # 왼쪽 → +y
    assert out.vy == pytest.approx(c.vy_max, abs=0.002) and abs(out.vx) < 1e-9
    out = settle(Mapper(c), Sample(0, 512, 0))  # 오른쪽 → -y
    assert out.vy < -0.9 * c.vy_max


def test_sign_options_flip_axes():
    out = settle(Mapper(Config()), Sample(1023, 0, 0))
    assert out.vx > 0 and out.vy > 0
    out = settle(Mapper(Config(fwd_sign=1, side_sign=-1)), Sample(1023, 0, 0))
    assert out.vx < 0 and out.vy < 0


def test_deadzone_and_diagonal_not_faster_than_max():
    c = Config()
    out = settle(Mapper(c), Sample(512 + 40, 512 - 40, 0))  # 12% 안쪽
    assert (out.vx, out.vy) == (0.0, 0.0)
    out = settle(Mapper(c), Sample(0, 0, 0))  # 대각선 끝: 벡터 크기가 1을 넘지 않는다
    assert math.hypot(out.vx / c.vx_max, out.vy / c.vy_max) <= 1.0 + 1e-9


def test_turn_buttons():
    c = Config()
    out = settle(Mapper(c), Sample(512, 512, bit(c.btn_left)))
    assert out.wz == pytest.approx(c.wz_max)
    out = settle(Mapper(c), Sample(512, 512, bit(c.btn_right)))
    assert out.wz == pytest.approx(-c.wz_max)
    out = settle(Mapper(c), Sample(512, 512, bit(c.btn_left) | bit(c.btn_right)))
    assert out.wz == 0.0


def test_acceleration_limit():
    c = Config()
    m = Mapper(c)
    out = run(m, [Sample(512, 0, 0)] * 4, dt=0.02)  # 3 tick 동안 0.5 m/s² × 0.06 s
    assert out.vx == pytest.approx(c.accel_v * 0.06)


def test_pot_scales_speed():
    c = Config(use_pot=True)
    out = settle(Mapper(c), Sample(512, 0, 0, 0))
    assert out.vx == pytest.approx(c.vx_max * c.pot_min)
    out = settle(Mapper(c), Sample(512, 0, 0, 1023))
    assert out.vx == pytest.approx(c.vx_max)


def test_height_buttons_step_hold_and_clamp():
    c = Config()
    m = Mapper(c)
    out = run(m, [CENTER, Sample(512, 512, bit(c.btn_up)), CENTER])  # 한 번 눌렀다 뗌
    assert out.height == pytest.approx(c.height + c.h_step)
    out = run(m, [Sample(512, 512, bit(c.btn_down)), CENTER], t0=1.0)
    assert out.height == pytest.approx(c.height)
    # 누르고 있으면 h_hold 뒤부터 계속 올라가고, 범위를 넘지 않는다
    out = run(m, [Sample(512, 512, bit(c.btn_up))] * 400, t0=2.0)
    assert out.height == pytest.approx(c.h_max)


def test_short_press_in_one_tick_is_not_lost():
    c = Config()
    m = Mapper(c)
    m.feed(CENTER)
    m.tick(0.0, True)
    m.feed(Sample(512, 512, bit(c.btn_up)))  # tick 사이에 눌렀다 뗌
    m.feed(CENTER)
    assert m.tick(0.02, True).height == pytest.approx(c.height + c.h_step)


def test_stick_short_press_resets_height_long_press_requests_reset():
    c = Config()
    m = Mapper(c)
    run(m, [CENTER, Sample(512, 512, bit(c.btn_up)), CENTER, Sample(512, 512, bit(c.btn_up)), CENTER])
    assert m.height == pytest.approx(c.height + 2 * c.h_step)
    out = run(m, [Sample(512, 512, bit(c.btn_stick)), CENTER], t0=1.0)  # 짧게 → 높이 기본값, 리셋 아님
    assert out.height == pytest.approx(c.height) and not out.reset

    m.feed(Sample(512, 512, bit(c.btn_stick)))  # 길게
    resets = [m.tick(3.0 + i * 0.02, True).reset for i in range(100)]  # 2초 동안 누름
    m.feed(CENTER)
    out = m.tick(5.1, True)
    assert sum(resets) == 1  # 한 번만
    assert not out.reset and out.height == pytest.approx(c.height)  # 길게 눌렀으면 뗄 때 높이를 건드리지 않음


def test_estop_toggles_and_zeroes_velocity():
    c = Config()
    m = Mapper(c)
    settle(m, Sample(512, 0, 0))
    out = run(m, [Sample(512, 0, bit(c.btn_estop)), Sample(512, 0, 0)], t0=3.0)
    assert out.estop and out.vx == 0.0
    out = run(m, [Sample(512, 0, 0)] * 10, t0=4.0)  # 켜져 있는 동안 계속 0
    assert out.estop and out.vx == 0.0
    out = run(m, [Sample(512, 512, bit(c.btn_estop)), CENTER], t0=5.0)
    assert not out.estop


def test_dead_joystick_stops_and_rebaselines_buttons():
    c = Config()
    m = Mapper(c)
    settle(m, Sample(512, 0, 0))
    out = m.tick(10.0, alive=False)
    assert out.mode == Mode.STOP and (out.vx, out.vy, out.wz) == (0.0, 0.0, 0.0)
    # 다시 연결됐을 때 이미 눌려 있던 버튼은 '눌림'이 아니다 (비상정지가 뒤집히지 않는다)
    out = run(m, [Sample(512, 512, bit(c.btn_estop))] * 3, t0=11.0)
    assert not out.estop and out.mode == Mode.POLICY


def test_no_samples_yet_is_stop_via_alive_false():
    m = Mapper(Config())
    assert m.tick(0.0, alive=False).mode == Mode.STOP


# ── 시리얼 읽기 ──
class FakeSerial:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.closed = False

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks and isinstance(self.chunks[0], bytes) else 0

    def read(self, n):
        if self.chunks and isinstance(self.chunks[0], Exception):
            raise self.chunks.pop(0)
        return self.chunks.pop(0) if self.chunks else b""

    def close(self):
        self.closed = True


def test_serial_source_reads_alive_and_reconnects():
    ports = [FakeSerial([b"<1,2,3,4>\n", OSError("usb gone")]), FakeSerial([b"<9,9,0,0>\n"])]
    opened = []

    def opener(p):
        opened.append(p)
        return ports.pop(0)

    src = SerialSource("/dev/fake", opener=opener)
    assert not src.alive(0.0)
    assert src.poll(100.0) == [Sample(1, 2, 3, 4)] and src.alive(100.1) and not src.alive(101.0)
    assert src.poll(100.2) == [] and src.ser is None  # 읽다가 예외 → 닫고 대기 (예외를 밖으로 내지 않음)
    assert src.poll(100.3) == [] and len(opened) == 1  # 재시도 간격 전
    assert src.poll(101.5) == [] and len(opened) == 2  # 다시 열림
    assert src.poll(101.6) == [Sample(9, 9, 0, 0)]


def test_serial_source_first_open_failure_exits():
    def opener(p):
        raise OSError("no such device")

    with pytest.raises(SystemExit):
        SerialSource("/dev/nope", opener=opener)


def test_real_serial_over_pty():
    """진짜 pyserial(timeout=0)이 pty에서 논블로킹으로 읽히는지."""
    serial = pytest.importorskip("serial")
    pty = pytest.importorskip("pty")
    master, slave = pty.openpty()
    try:
        src = SerialSource(os.ttyname(slave))
        assert isinstance(src.ser, serial.Serial)
        assert src.poll(time.monotonic()) == []  # 데이터 없음 → 기다리지 않고 바로 돌아옴
        os.write(master, b"<512,0,1,800>\n<51")
        time.sleep(0.05)
        got = src.poll(time.monotonic())
        assert got == [Sample(512, 0, 1, 800)]
        src.close()
    finally:
        os.close(master)
        os.close(slave)


# ── 프로토콜 끝에서 끝까지 ──
def test_end_to_end_to_command_receiver():
    c = Config()
    port = 15598
    rx = CommandReceiver(port=port, default_height=0.2, timeout=0.5)
    tx = CommandSender("127.0.0.1", port)
    m = Mapper(c)
    try:
        t = 0.0
        out = None
        for _ in range(60):  # 위로 끝까지 밀고 1.2초
            m.feed(Sample(512, 0, 0))
            out = m.tick(t, True)
            tx.send(out.vx, out.vy, out.wz, out.height, out.mode, out.estop)
            t += 0.02
        time.sleep(0.02)
        a = rx.poll()
        assert a.mode == Mode.POLICY and not a.clamped
        assert a.vx == pytest.approx(c.vx_max, abs=1e-5) and a.vy == pytest.approx(0.0)

        out = m.tick(t, alive=False)  # 조이스틱이 끊김 → STOP 명령
        tx.send(out.vx, out.vy, out.wz, out.height, out.mode, out.estop)
        time.sleep(0.02)
        a = rx.poll()
        assert a.mode == Mode.STOP and a.vx == 0.0
    finally:
        tx.close()
        rx.close()


# ── 파이 노드(WebSocket) 조종: scripts/teleop/joystick_ws.py ──
def _out(vx=0.0, vy=0.0, wz=0.0, estop=False, tap=False):
    from joystick_sender import Output
    return Output(vx, vy, wz, 0.2, Mode.POLICY, estop, False, tap)


def test_ws_walk_only_when_moving_and_stop_once():
    from joystick_ws import WsCommander
    c = WsCommander()
    assert c.messages(_out(), 0.0) == []  # 가만히 있을 때는 아무것도 안 보낸다 (0 속도 walk는 서보를 켠다)
    assert c.messages(_out(vx=0.1), 0.05) == [{"cmd": "walk", "vx": 0.1, "vy": 0.0, "wz": 0.0}]
    assert c.messages(_out(vx=0.1, wz=-0.3), 0.1)[0]["wz"] == -0.3
    assert c.messages(_out(), 0.15) == [{"cmd": "stop"}]  # 멈추면 한 번
    assert c.messages(_out(), 0.2) == []


def test_ws_tap_is_stand():
    from joystick_ws import WsCommander
    assert WsCommander().messages(_out(tap=True), 0.0) == [{"cmd": "stand"}]


def test_ws_estop_never_sends_walk_and_resumes_with_stand():
    from joystick_ws import ESTOP_RESEND, WsCommander
    c = WsCommander()
    c.messages(_out(vx=0.1), 0.0)
    assert c.messages(_out(vx=0.1, estop=True), 0.1) == [{"cmd": "estop"}]
    assert c.messages(_out(vx=0.1, estop=True, tap=True), 0.2) == []  # 중간에는 walk·stand 모두 안 나간다
    assert c.messages(_out(estop=True), 0.1 + ESTOP_RESEND) == [{"cmd": "estop"}]  # 가끔 다시 알린다
    assert c.messages(_out(), 2.0) == [{"cmd": "stand"}]  # 풀면 일어선다
    assert c.messages(_out(), 2.1) == []


def test_ws_reconnect_resends_estop_and_forgets_moving():
    from joystick_ws import WsCommander
    c = WsCommander()
    c.messages(_out(estop=True), 0.0)
    c.reconnected()
    assert c.messages(_out(estop=True), 0.1) == [{"cmd": "estop"}]
    c = WsCommander()
    c.messages(_out(vx=0.1), 0.0)
    c.reconnected()
    assert c.messages(_out(), 0.1) == []  # 새 연결에는 보낼 stop이 없다


def test_pilink_against_real_websocket_server():
    """진짜 websockets 서버에 붙어 보낸 명령이 도착하고, 서버가 죽었다 살아나면 다시 붙는지."""
    import json
    import threading

    pytest.importorskip("websockets")
    from joystick_ws import PiLink
    from websockets.sync.server import serve

    got = []

    def handler(ws):
        ws.send(json.dumps({"type": "hello", "robot": "x", "version": "0", "simulated": True}))
        ws.send(json.dumps({"type": "telemetry", "mode": "stand", "safety": [], "robot": {"foot": [1] * 6}}))
        for raw in ws:
            got.append(json.loads(raw))

    with serve(handler, "127.0.0.1", 0) as server:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        port = server.socket.getsockname()[1]
        link = PiLink(f"ws://127.0.0.1:{port}", token="tok")
        t0 = time.monotonic()
        assert link.poll(t0) is True  # 첫 연결
        link.send({"cmd": "walk", "vx": 0.1, "vy": 0.0, "wz": 0.0}, t0)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and (link.tel_time < 0 or len(got) < 2):
            link.poll(time.monotonic())
            time.sleep(0.02)
        assert link.tel["mode"] == "stand"
        assert got[0] == {"cmd": "auth", "token": "tok"} and got[1]["cmd"] == "walk"
        link.close()


def test_pilink_connect_failure_does_not_raise_and_retries():
    from joystick_ws import PiLink
    calls = []

    def connect(url):
        calls.append(url)
        raise OSError("refused")

    link = PiLink("ws://nowhere:1", None, connect=connect)
    assert link.poll(0.0) is False and link.ws is None and len(calls) == 1
    link.send({"cmd": "stop"}, 0.1)  # 연결이 없어도 예외 없음
    assert link.poll(0.5) is False and len(calls) == 1  # 재시도 간격 전
    assert link.poll(1.5) is False and len(calls) == 2
