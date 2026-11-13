# 로봇 런타임 (Raspberry Pi)

통합 정책을 실물에서 돌리기 위한 부품. numpy + onnxruntime만 쓴다 (센서 노드는 ROS2 추가).

| 파일 | 내용 |
|---|---|
| `robot_policy.py` | `RobotPolicy`: 센서 값 → student 관측(282) → 단계 신경망 → student → 관절 목표각. 관절각 자리에 **마지막 목표각**을 쓴다 |
| `hexapod_kinematics.py` | URDF 순기구학. 목표각으로 몸통 기준 발끝 높이(`feet_below_body`)를 계산 |
| `phase_rule.py` | `PhaseRule`: 규칙 단계 추정기 (`mdp/phase_estimator.py` + `check_phase`). 단계 신경망 입력 `phase_rule`(40)·`phase_timer`(1). 표와 기준은 `robot_io.json`의 `phase_rule`에서 읽는다 |
| `joint_map.py` | sim 관절각 [rad] ↔ 실물 '실제 관절 각도' [도] (HexaCore `leg_model` 정의). hip = 180 ± coxa (왼 +, 오른 −), femur = 270 − femur, tibia = 268.44 + tibia |
| `pose_check.py` | 1단계 실물 검사: 다리를 공중에 띄우고 sim 기본 자세로 옮긴 뒤 관절을 하나씩 움직여 다리 번호·관절·방향을 눈으로 확인 |
| `l2_sensors.py` | `L2Fusion`: L2 IMU 각속도·Point-LIO 자세·월드 점군 → 몸통 각속도, 중력 방향, 41×5 높이맵, `body_x`. `main()`은 ROS2 확인용 노드 |
| `test_runtime.py` | 위 셋을 시뮬레이션 기준 기록·합성 계단으로 검사 |

높이맵은 측정 도구와 같은 `scripts/lidar_error/heightmap.py`로 만든다.

## 검사

```bash
uv run deploy/runtime/test_runtime.py deploy/mixed_phase_v1
```

2026-11-14 결과: 순기구학 1.6e-7 m, 관측·logits·행동 1e-5 이내, 규칙 단계 4000 step 모두 일치, 규칙 특징 1.2e-7,
높이맵 모서리 밖 0 cm, body_x·계단 좌표 몸통 0.8 cm, 각속도·중력 1e-16.

## 한 step 순서 (50 Hz)

```python
s = fusion.snapshot()                              # L2Fusion
pose = fusion.stairs_pose(policy.x0)               # 계단 좌표 몸통. 수직면을 못 찾았으면 계단 3 m 앞 평지로 본다
r = Readings(s["gyro"], s["gravity"], contacts, s["height_map"], command, s["body_x"])
out = policy.step_with_rule(r, *pose, fusion.speed())   # 측정 → 단계 진행 → 규칙 특징 → 신경망
send(out.target)
```

`policy.reset(start_move)`은 동작을 시작할 때 부른다. 계단 앞(앞발이 수직면 뒤)에서 시작하면 0이다.

## 1단계: 관절 변환 확인 (다리를 공중에 띄우고)

```bash
python3 deploy/runtime/pose_check.py --hexacore ~/HexaCore      # 로봇 Pi, ESP32 = /dev/ttyAMA0
```

HexaCore 서기 자세 → 3초 동안 sim 기본 자세(허벅지 70° 아래, 종아리 거의 수직) → `all`로 18관절을 하나씩 sim +10° 움직이며
화면의 "보여야 하는 움직임"과 비교해 y/n을 적는다. 결과는 `pose_check_log.jsonl`. 서보 없이 계산만 하려면 `--sim`.

## 로봇에서 센서 확인 (ROS2)

1. `unitree_lidar_ros2` (unilidar_sdk2): 작업 모드 0 = 표준 FOV, 3D, **IMU 켬**(bit2 = 0), 이더넷, 자동 시작.
2. Point-LIO (`point_lio_unilidar`, L2 설정, `gravity_align` 켬).
3. 확인용 노드 (`base_in_imu`는 CAD에서 읽은 L2 IMU 좌표의 몸통 원점 `x y z [m] roll pitch yaw [도]`):

```bash
python3 deploy/runtime/l2_sensors.py --base_in_imu X Y Z ROLL PITCH YAW
```

1초마다 각속도, 중력 방향, body_x, 높이맵 일부를 출력한다. IMU 토픽이 2초 동안 안 오면 작업 모드를 확인하라고 경고한다.

## 알아 둘 점

- **관절각 = 목표각 모델이 필요하다.** 실제 관절각으로 학습한 `mixed_phase_v1`에 목표각을 넣으면 시뮬레이션에서 계단 0%, 걷기 20초 유지 0.3%였다.
  목표각으로 이어 학습하는 task가 `Hexapod-Student-A-Phase-Cmd`이고, 그 결과로 단계 신경망을 다시 학습한 뒤 새 폴더로 내보내야 한다.
- **높이맵 모서리:** `heightmap.py`는 지도를 3×3칸(3 cm) 평균으로 읽어서, 수직면 바로 앞 한 줄은 아래·윗단이 섞인다 (합성 계단에서 최대 8 cm = 반 단).
  시뮬레이션 높이맵은 날카로운 모서리다.
- **L2 IMU 단위:** 각속도는 rad/s로 가정했다 (공식 문서에 단위가 없다). 로봇을 손으로 90° 돌려 적분값을 확인할 것.

## 아직 없는 것

- 계단이 안 보이는 평지에서는 "계단이 3 m 앞에 있다"고 넣는다 (학습 걷기 env와 비슷한 값). 평지를 걷다가 계단이 처음 보이는 순간
  body_x가 -3 m에서 실제 값으로 뛰는데, 이런 전환은 학습에 없었다. 실물에서는 계단 앞에서 멈춘 뒤 `reset(0)`으로 계단 모드를 시작하는 편이 안전하다.
- 50 Hz 정책 루프: 발끝 스위치(`#ft`)·조종 명령(HXCMD) 읽기, 목표각 → 펌웨어 각도 변환(`HexaCore` `leg_model.to_firmware`), 다리 번호·관절 영점·방향 확인.
