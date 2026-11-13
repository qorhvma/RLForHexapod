# 걷기·계단 통합 정책 v1 (실물 로봇용 파일)

| 항목 | 값 |
|---|---|
| student | `logs/rsl_rl/hexapod_student/2026-11-13_19-58-44_A_phase_mixed/model_1499.pt` (task `Hexapod-Student-A-Phase`) |
| 단계 추정 신경망 | `logs/phase_net/2026-11-13_22-27-48/model_1499.pt` |
| teacher | 걷기: `logs/teachers.pt`의 walk, 계단: `hexapod_stair/2026-11-10_17-20-12/model_4250.pt` |
| 시뮬레이션 성능 (센서 오차 없음, 2048대) | 계단 꼭대기 도달 92.7%, 걷기 20초 유지 100%, 속도 오차 xy 0.048 m/s |
| 참고: 단계 참값을 넣으면 | 계단 98.9% (teacher 98.8%) |

만든 방법 (Isaac Lab PC):

```bash
uv run python scripts/export_robot_policy.py --checkpoint <student.pt> --phase_net <phase_net.pt> --out deploy/mixed_phase_v1
```

검사 (아무 PC, 로봇에서도 가능. GPU·Isaac Lab 필요 없음):

```bash
uv run scripts/verify_robot_policy.py deploy/mixed_phase_v1
```

2026-11-14 결과: logits 최대 차이 1.1e-5, actions 최대 차이 7.7e-6, 단계 칸 4000 step 모두 일치 → 통과.

## 파일

| 파일 | 내용 |
|---|---|
| `student.onnx` | `obs (1, 282)`, `h_in (1, 1, 256)` → `actions (1, 18)`, `h_out`. 관측 정규화가 안에 들어 있어 원래 값을 그대로 넣는다 |
| `phase_net.onnx` | `x (1, 326)`, `h_in (1, 128)` → `logits (1, 7)`, `h_out`. 정규화 포함 |
| `robot_io.json` | 관측 항의 순서·크기·뜻, 행동 → 관절 목표각 식, 관절 이름·기본 자세·범위, 높이맵 격자, 단계 이름 |
| `reference.npz` | 시뮬레이션 기록 500 step × 8대 (계단 4, 걷기 4): 관측, 신경망 입력, logits, actions, 참값 단계 |

## 매 step (50 Hz) 할 일

1. 센서로 아래 관측을 만든다.
2. `phase_net.onnx`: `x` = student 관측에서 단계 칸을 뺀 것(275) + `body_x`(4) + `feet_below_body`(6) + `rule`(40) + `time`(1).
3. `phase = one_hot(argmax(logits))`를 student 관측의 단계 칸(`[275:282]`)에 넣는다.
4. `student.onnx`를 돌린다. `h_out`은 다음 step의 `h_in`이 된다. 동작을 새로 시작할 때는 두 신경망 모두 `h_in = 0`.
5. `target = default + scale × clip(actions, -2, 2)`, 관절 범위로 자른다 (값은 `robot_io.json`의 `action`).
6. 다음 step의 `last_action`에는 자르기 **전** `actions`를 넣는다.

계단은 명령 `(0.2, 0, 0, 0.24)`로만 학습했다. 계단을 오를 때는 이 명령을 준다.

## 실물에서 아직 만들 수 없는 입력

로봇 쪽 코드(`HexaCore-clone/software/raspberrypi`) 기준 (2026-11-14).

| 관측 | 크기 | 필요한 것 | 지금 로봇 | 상태 |
|---|---|---|---|---|
| `joint_pos` | 18 | 관절각 측정 | PWM 서보라 읽는 경로 없음 (`ServoPotSweep`으로 가변저항 실험만 함) | **없음** |
| `joint_vel` | 18 | 관절 각속도 | 위와 같음 | **없음** |
| `base_ang_vel` | 3 | 자이로 | ESP32 `#imu`는 roll·pitch뿐. L2 내장 IMU에 자이로가 있으나 기본 꺼짐, `decode_imu_packet`은 오프셋 미확인 | **없음** |
| `projected_gravity` | 3 | 중력 방향 | `#imu` roll·pitch로 계산 가능 | 가능 |
| `last_action` | 18 | 직전 행동 | 정책 실행기가 기억 | 가능 |
| `feet_contact` | 6 | 발끝 스위치 | `#ft` (순서·부호 확인 필요) | 가능 |
| `height_map` | 205 | 몸통 앞 지형 높이 격자 | L2 점 디코더가 틀만 있음 (`decode_point_packet` TODO), 위치 추정은 오도메트리뿐 | **없음** |
| `command` | 4 | 조종 명령 | 조이스틱 → WebSocket (`joystick_ws.py`) | 가능 |
| `body_x` (신경망) | 4 | 첫 수직면 기준 몸통 x | 수직면 검출 + 위치 추정 필요 | **없음** |
| `feet_below_body` (신경망) | 6 | 발끝 높이 (IMU + 관절각 FK) | 관절각이 없음 | **없음** |
| `rule`, `time` (신경망) | 41 | 규칙 단계 추정기 (`mdp/phase_estimator.py`) | 로봇용 코드 없음. 계단 위치·높이도 필요 | **없음** |

행동 쪽도 확인이 남았다.

- 시뮬레이션 관절 이름(L1~L6)과 펌웨어 다리 번호(0~5)의 대응. 펌웨어 쪽 가정은 "0 왼앞, 1 왼가운데, 2 왼뒤, 3 오른뒤, 4 오른가운데, 5 오른앞"이고, 시뮬레이션은 L3 왼앞, L2 왼가운데, L1 왼뒤, L6 오른뒤, L5 오른가운데, L4 오른앞이다. 둘 다 실물로 확인하지 않았다.
- 시뮬레이션 관절각(femur 기본 20°, tibia 기본 −110°)과 실제 관절 각도(180 = 일자로 편 자세, `calibration.json`)의 변환. 영점과 방향을 실물로 맞춰야 한다. 그 뒤 펌웨어 각도로 바꾸는 것은 `leg_model.to_firmware`가 한다.
- 시뮬레이션의 서보 모델(토크·속도 한계)이 실물 RC920DMG·MG996R과 얼마나 다른지는 측정하지 않았다.
