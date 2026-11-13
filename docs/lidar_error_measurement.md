# 라이다 오차 측정

실물 라이다(Unitree L2 + Point-LIO)가 policy에 주는 값(41×5 높이맵, 첫 수직면 기준 몸통 x)을 참값과 비교해서
`sensor_model.SensorErrorCfg`의 라이다 항목을 정한다. 코드는 `scripts/lidar_error/`에 있다.

| 파일 | 역할 | 도는 곳 |
|---|---|---|
| `heightmap.py` | 지도 쌓기, 높이맵 자르기, 수직면 찾기 (numpy만 씀, 시뮬레이션과 같은 정의) | 로봇, PC |
| `record_ros2.py` | Point-LIO 토픽을 받아 trial 파일(`.npz`) 저장 | 로봇 (ROS2) |
| `analyze.py` | trial들 → 항목별 측정값, 학습 최댓값 제안, 지금 학습 값과 비교 | PC |
| `synth.py` | 오차를 넣은 합성 trial (분석 확인, 보고서 미리 보기) | PC |

`heightmap.py`의 지도 쌓기는 실물에 올릴 높이맵 코드와 같아야 측정이 의미가 있다. 배포용 높이맵도 이 모듈로 만들어야 한다.

## 준비

1. **Point-LIO (ROS2)**: `unitree_lidar_ros2`로 L2 점군과 IMU를 받고, Point-LIO는 `gravity_align`을 켠다.
   토픽 기본값은 `/aft_mapped_to_init`(자세)와 `/cloud_registered`(월드 좌표 스캔)이며, 다르면 `--odom_topic`, `--cloud_topic`으로 바꾼다.
2. **`--base_in_lio`**: Point-LIO 몸통(L2 IMU) 좌표에서 본 로봇 몸통 원점(URDF base)의 위치와 자세 `x y z [m] roll pitch yaw [deg]`.
   CAD에서 L2 IMU 위치를 읽어 넣는다. URDF의 라이다 링크 원점은 메시 좌표라서 IMU 좌표와 다를 수 있다.
3. **계단 실측**: 단마다 높이와 깊이를 재서 json으로 만든다 (계단 좌표: 첫 수직면 x=0, 단 윗면 높이는 바닥 기준).
   ```json
   {"riser_x": [0.0, 0.301, 0.598], "top_z": [0.161, 0.320, 0.482], "end_x": 1.40, "width": 1.0}
   ```
4. **참값 표시**: 바닥에 첫 수직면에서 -0.30, -0.15, 0, +0.15, +0.30 m 떨어진 선을 테이프로 붙인다.
   기준점은 몸통 원점을 바닥에 내린 점이다. 로봇은 계단 가운데 선 위에 두고, 계단을 똑바로 바라보게 놓는다.
5. **받침대**: static 측정은 단단한 받침대 위에서 한다. 몸통 높이(`--true_z`, 바닥에서 몸통 원점까지)는 자로 재고,
   수평은 디지털 수평계로 확인한다. 다리 자세 오차가 섞이지 않게 하기 위해서다.

## trial 목록

trial마다 **Point-LIO를 새로 켠다.** 켤 때마다 달라지는 편향이 `lidar_x_bias`(에피소드 고정 오차)에 해당하기 때문이다.

| 종류 | 위치 | 반복 | 시간 | 재는 것 |
|---|---|---|---|---|
| static, 계단 앞 | true_x = -0.30, -0.15 | 각 4 | 30 s | x 편향·노이즈, 높이 편향, 기울기, 칸 노이즈, 구멍, 주기 |
| static, 계단 위 (받침대를 단 위에) | true_x = 0, +0.15, +0.30 | 각 4 | 30 s | 같음 + 단 위에서의 수직면 찾기 |
| static, 평지 (모서리가 높이맵 밖) | true_x = -1.0 | 3 | 30 s | 기울기, 높이 편향 (모서리 영향 없이) |
| walk | -0.40 출발 → 꼭대기에서 멈춤 | 3 이상 | 시작·끝 3 s 정지 | 걷는 동안 쌓이는 x 드리프트 |

```bash
python3 scripts/lidar_error/record_ros2.py --name s_m030_r01 --kind static --true_x -0.30 --true_z 0.17 --stairs stairs.json --base_in_lio 0 0 0 0 0 0 --duration 30
```

```bash
python3 scripts/lidar_error/record_ros2.py --name w_r01 --kind walk --true_x -0.40 --true_x_end 0.95 --true_z 0.17 --stairs stairs.json --base_in_lio 0 0 0 0 0 0 --duration 60
```

`--base_in_lio`는 위 2번에서 구한 값으로 바꾼다. 원시 데이터(`/unilidar/cloud`, `/unilidar/imu`)도 `ros2 bag record`로 같이 남겨 두면,
나중에 Point-LIO 설정을 바꿔 다시 돌린 뒤 `ros2 bag play`와 `record_ros2.py`로 trial을 새로 만들 수 있다.

## 분석

```bash
python3 scripts/lidar_error/analyze.py lidar_trials/ --json lidar_error_summary.json
```

앞 3초(`--warmup`)는 지도가 쌓이는 동안이라 버린다. 출력되는 내용:

- **trial별 표**: x 편향(평균), x 노이즈(표준편차), rho1(노이즈가 프레임끼리 이어지는 정도), 지도 밀림, 높이 편향, pitch·roll,
  칸 노이즈, 모서리 어긋남 비율, 구멍(갱신 안 된 칸) 비율, 안 보이는 칸 비율, 갱신 주기
- **학습 설정 제안**: trial들의 95% 값 × `--margin`(1.5)을 학습 최댓값으로 제안하고, `train_phase_net.py`의 `TRAIN_ERRORS`와
  `env_cfg.py`의 student 높이맵 노이즈 std와 비교한다.
  x 편향은 "정지 편향 + m당 드리프트 × `--climb_length`(1.2 m)"로 계산한다.
- **map_mode**: trial별 지도 밀림이 몸통 x 편향과 같으면 `accumulated`(지금 설정), 0이면 `per_frame`.

반영할 곳:

| 측정 항목 | 반영할 곳 |
|---|---|
| `lidar_x_bias` … `lidar_dropout` | `train_phase_net.py`의 `TRAIN_ERRORS`, A-PhaseNoise env 설정 |
| `height_noise_std` | `env_cfg.py` student `height_map`의 `GaussianNoiseCfg(std)` |
| `lidar_period` | 실물은 L2 스캔 주기다 (학습 기본값 0은 매 step 갱신). 평가(`eval_stairs.py`)에 측정값을 넣어 성공률을 확인한다 |

"키울 것"이 뜬 항목이 있으면 그 항목만 키워서 phase net과 student를 다시 학습한다.

## 미리 보기, 시험

```bash
python3 scripts/lidar_error/synth.py --out /tmp/lidar_synth
```

```bash
python3 scripts/lidar_error/analyze.py /tmp/lidar_synth
```

```bash
python3 -m pytest tests/test_lidar_error.py -q
```

## 한계

- **지도 밀림은 ±1 cm까지만 보인다.** 높이맵 격자가 2 cm라서다. map_mode 판정은 x 편향이 5 mm보다 클 때만 의미가 있다.
  이 파이프라인은 높이맵과 몸통 x를 같은 지도와 같은 자세에서 내므로, 구조상 `accumulated`가 맞다.
- **walk는 시작과 끝만 참값이 있다.** 걷는 중의 진동으로 커지는 높이맵 노이즈는 따로 재지 못한다.
- **지연은 시뮬레이션에 없다.** 수신 시각과 스캔 시각의 차이를 latency로 따로 보여 준다.
  ROS 시각이 센서 시각이거나 bag 재생이면 값이 맞지 않는다.
- **지도 쌓기 설정은 실물에서 조정이 필요할 수 있다.** 합성 점군(노이즈 3 mm)에서는 수직면 위치 오차가 ±5 mm 안이었다.
  실물 L2 노이즈가 크면 `TerrainMap`의 `band`(3 cm)와 `face_spread`(6 cm)를 키운다.
  이 값은 배포용 높이맵 설정과 같이 바꿔야 한다.
