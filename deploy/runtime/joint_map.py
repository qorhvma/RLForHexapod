"""시뮬레이션 관절각 [rad] ↔ 실물 '실제 관절 각도' [도] (HexaCore leg_model 정의).

    true = sim_to_true(target)     # target: (18,) robot_io.json joint_names 순서 → (6, 3) 펌웨어 다리 순서, [hip, femur, tibia]
    sim = true_to_sim(true)        # 역변환

실제 관절 각도 (HexaCore software/raspberrypi/leg_model.py 머리말):
    hip   : 180 = 다리가 중립 방향(mounts.heading_deg). 커지면 다리가 앞쪽으로 돈다 (양쪽 모두)
    femur : 180 = 수평, 작아질수록 위로 든다, 270 = 수직 아래
    tibia : femur와 tibia 사이 안쪽 각도, 180 = 일자

URDF 순기구학으로 기하를 계산해서 얻은 식 (test_runtime.py check_joint_map이 URDF로 다시 확인한다):
    hip   = 180 + side × sim_coxa      side = +1 왼쪽 다리, -1 오른쪽 다리
            (sim coxa +는 위에서 보면 모든 다리가 시계 방향 → 왼다리는 앞으로, 오른다리는 뒤로)
    femur = 270 - sim_femur            (sim femur 0 = 수직 아래, + = 들어 올림)
    tibia = TIBIA_OFS + sim_tibia      (sim tibia + = 펴짐)
TIBIA_OFS = 270 - 1.56: 발끝 추정점(TIP_OFFSET)이 tibia 관절축 선에서 1.56° 옆에 있어서, 무릎 → 발끝 선을 '일자'의 기준으로 삼았다.
"""

import math

import numpy as np

# 펌웨어 다리 번호 0~5 → URDF 다리 (HexaCore calibration.json kinematics.mounts의 urdf 칸과 같다)
FW_TO_URDF = ["L3", "L2", "L1", "L6", "L5", "L4"]  # 왼앞, 왼가운데, 왼뒤, 오른뒤, 오른가운데, 오른앞
FW_NAMES = ["왼앞", "왼가운데", "왼뒤", "오른뒤", "오른가운데", "오른앞"]
SIDE = [1, 1, 1, -1, -1, -1]
TIBIA_OFS = 270.0 - math.degrees(math.atan2(0.004, 0.147))  # = 268.44, hexapod_kinematics.TIP_OFFSET의 x, y
SIM_JOINTS = ("coxa", "femur", "tibia")


def _index(joint_names: list[str]) -> np.ndarray:
    """(6, 3) 표: [펌웨어 다리][hip, femur, tibia] → joint_names 안의 위치."""
    return np.array([[joint_names.index(f"{leg}_{j}") for j in SIM_JOINTS] for leg in FW_TO_URDF])


def sim_to_true(target_rad, joint_names: list[str]) -> np.ndarray:
    """sim 관절각 (18,) [rad] → 실제 관절 각도 (6, 3) [도], 펌웨어 다리 순서."""
    q = np.degrees(np.asarray(target_rad, float))[_index(joint_names)]
    side = np.array(SIDE, float)
    return np.stack([180.0 + side * q[:, 0], 270.0 - q[:, 1], TIBIA_OFS + q[:, 2]], axis=1)


def true_to_sim(true_deg, joint_names: list[str]) -> np.ndarray:
    """실제 관절 각도 (6, 3) [도] → sim 관절각 (18,) [rad], joint_names 순서."""
    t = np.asarray(true_deg, float)
    side = np.array(SIDE, float)
    q = np.stack([(t[:, 0] - 180.0) * side, 270.0 - t[:, 1], t[:, 2] - TIBIA_OFS], axis=1)
    out = np.zeros(len(joint_names))
    out[_index(joint_names)] = np.radians(q)
    return out


def describe(fw_leg: int, joint: int, sign: float) -> str:
    """sim 관절각을 +(sign>0) 또는 -로 움직였을 때 실물에서 보여야 하는 움직임."""
    up = sign > 0
    if joint == 0:
        fwd = up if SIDE[fw_leg] > 0 else not up
        return f"{'앞' if fwd else '뒤'}쪽으로 돈다 (위에서 보면 {'시계' if up else '반시계'} 방향)"
    if joint == 1:
        return f"femur(허벅지)가 {'위로 들린다' if up else '아래로 내려간다'}"
    return f"tibia(종아리)가 {'펴진다 (무릎 각도가 커짐)' if up else '접힌다 (무릎 각도가 작아짐)'}"

