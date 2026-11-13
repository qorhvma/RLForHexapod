# 옛 조이스틱으로 조종하기

강화학습 이전(`HexaCore-clone/legacy/bluetooth_pc`)에 쓰던 우노(오렌지보드) + 조이스틱 쉴드를 쓴다.
우노 펌웨어는 `joystick_controller.ino` 그대로이고 USB 시리얼 115200으로 `<x,y,버튼,볼륨>`을 50ms마다 보낸다.
조이스틱은 **조종하는 PC(노트북)에 꽂는다.** 받는 쪽(로봇 파이 또는 시뮬레이션)에 따라 스크립트가 둘이다.

```
                                       ┌─ joystick_ws.py ─ WebSocket :8765 ─▶ [파이 robot_main.py] ─ USB ─▶ [ESP32] (실물, 지금 가능)
[우노 + 조이스틱] ─USB 시리얼─▶ [노트북] ─┤
                                       └─ joystick_sender.py ─ UDP 15500 ─▶ [HXCMD 수신: 시뮬레이션 teleop.py / 나중의 파이 정책 실행]
```

| | 스크립트 | 프로토콜 | 받는 쪽 | 상태 |
|---|---|---|---|---|
| 실물 로봇 | [joystick_ws.py](../scripts/teleop/joystick_ws.py) | WebSocket + JSON (`walk`/`stop`/`stand`/`estop`) | `HexaCore-clone/software/raspberrypi/robot_main.py` (삼각 보행) | 가상 우노 + 파이 노드 `--sim`으로 종단 시험 |
| 시뮬레이션·RL 정책 | [joystick_sender.py](../scripts/teleop/joystick_sender.py) | HXCMD v1 (UDP) | `scripts/teleop/teleop.py`. 파이의 RL 정책 실행 코드는 아직 없음 | 가상 우노로 시험 |

둘 다 조이스틱 읽기·매핑은 같은 코드(`joystick_sender.py`의 `SerialSource`, `Mapper`)를 쓴다.

## 준비 (노트북)

```bash
pip install pyserial websockets    # websockets는 joystick_ws.py만. 11 이상
sudo usermod -aG dialout $USER     # 한 번만, 그 뒤 로그아웃·로그인 (포트 권한)
```

우노를 꽂고 먼저 방향과 버튼 번호를 확인한다. 포트는 한 프로그램만 쓸 수 있으므로 확인이 끝나면 Ctrl+C로 끈다.

```bash
python3 scripts/teleop/joystick_ws.py --joytest
```

## 실물 로봇 (파이 노드)

1. **파이**: 와이파이(또는 Tailscale)에 붙이고 IP를 확인한다 (`hostname -I`). 노트북과 같은 네트워크여야 한다.
2. **파이**: 파이 노드를 켠다. 최초 설치는 `pip install -r requirements.txt`.
   ```bash
   cd HexaCore-clone/software/raspberrypi && python3 robot_main.py
   ```
3. **노트북**: 조이스틱 프로그램을 켠다.
   ```bash
   python3 scripts/teleop/joystick_ws.py --host <파이 IP>
   ```
4. 화면 줄에 `[stand] 발 ●●●●●● ... ms`가 보이면 연결된 것이다. 조이스틱을 **짧게 눌러 서고**, 기울여서 걷는다.

| 입력 | 동작 |
|---|---|
| 조이스틱 | 이동. 최고 속도는 노드의 한계 vx 0.12, vy 0.08 m/s |
| Left / Right (버튼 4 / 5) | 왼쪽 / 오른쪽 회전 (wz 0.6 rad/s) |
| 조이스틱 짧게 누름 (버튼 6) | 일어서기 (`stand`: 앉은 높이에서 몸을 서기 높이 80mm 까지 들어 올림) |
| 노랑 / 파랑 (버튼 0 / 1) | 몸 높이 올림 / 내림 (`height`: 한 번 10mm, 누르고 있으면 초당 40mm) |
| 버튼 2 | 비상정지 켜기 (`estop`: 서보 힘 빼기, 로봇이 주저앉는다) / 끄기 (`stand`로 다시 일어선다) |

길게 누름 리셋은 이 노드에 해당 명령이 없어서 쓰지 않는다.
파이 노드에 `HEXAPOD_TOKEN`을 설정했다면 `--token` (또는 같은 환경변수)을 준다.

안전:
- 움직일 때만 `walk`를 보낸다. 0 속도 `walk`도 노드를 서기 모드로 바꿔 서보를 켜 버리기 때문이다. 멈추면 `stop`을 한 번 보낸다.
- 비상정지 중에는 `walk`를 절대 보내지 않는다 (노드는 `walk`를 받으면 비상정지를 풀고 일어선다).
- 노드는 0.5초 안에 `walk`가 안 오면 스스로 멈춘다 (데드맨). 이 프로그램이 죽거나 네트워크가 끊겨도 같다.
- 조이스틱 신호가 0.5초 없거나 USB가 빠지면 정지하고, 다시 꽂으면 이어 붙는다. 이때 이미 눌려 있던 버튼은 눌림으로 세지 않는다.

## 시뮬레이션 / HXCMD

```bash
python3 scripts/teleop/joystick_sender.py --host <받는 쪽 IP>     # 기본 127.0.0.1
```

원격 Windows PC에서 `uv run python scripts/teleop/teleop.py --viz newton_gl`을 켜 두고 그 IP로 보낸다.
Windows 방화벽에서 UDP 15500 인바운드를 허용해야 한다. 버튼 동작은 아래만 다르다.

| 입력 | 동작 |
|---|---|
| 노랑 / 파랑 (버튼 0 / 1) | 몸 높이 올림 / 내림 (한 번 `--h_step`, 누르고 있으면 계속) |
| 조이스틱 짧게 / 1초 이상 | 높이 기본값 / 리셋 요청 |
| 버튼 2 | 비상정지 켜기 / 끄기 (HXCMD estop 플래그) |

받는 쪽의 안전 규칙(타임아웃 0.25 s, 학습 범위로 자르기, 첫 보낸 주소만 따르기)은 [protocol.py](../scripts/teleop/protocol.py) 그대로다.

## 공통 조정

버튼 번호·방향은 옛 `hexapod_joy_smooth.py`의 값이다. 다르면 `--joytest`로 눌러 보고
`--btn_up/down/estop/left/right/stick`, `--fwd_sign`, `--side_sign`, `--center`, `--deadzone`으로 맞춘다.
버튼 비트는 우노 펌웨어 기준 A=0, B=1, C=2, D=3, Left=4, Right=5, 조이스틱 누름=6이다.
스틱을 놓았을 때 값이 512에서 벗어나 있으면(실측 x≈437, y≈499) 가만히 있어도 조금 흐를 수 있다. `--deadzone 0.16` 등으로 키운다.

## 확인 상태

- `tests/test_joystick_sender.py` 25개: 파싱, 매핑, 가짜 시리얼 재연결, pty 위의 pyserial, `CommandReceiver`까지의 종단,
  WebSocket 명령 로직(움직일 때만 `walk`, 비상정지 중 침묵, 해제 시 `stand`), 진짜 websockets 서버 접속.
- 가상 우노(pty) + 실제 `robot_main.py --sim`으로 종단 시험: 정지 중 무송신, 전진 vx 0.12, 회전 wz 0.6, 놓으면 stand,
  비상정지 중 전진 무시, 해제 후 재개.
- **실제 우노 조이스틱 + 실제 파이 + 실물 로봇 조합은 아직 해 보지 않았다.** 처음에는 로봇을 들어 올린 상태나 다리가 바닥에 닿지 않는 상태에서 시험할 것.
