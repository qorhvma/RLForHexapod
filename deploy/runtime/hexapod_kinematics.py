"""몸통 기준 발끝 위치 (numpy 순기구학). 시뮬레이션과 같은 URDF, 같은 발끝 추정점을 쓴다.

    kin = HexapodKinematics()                       # src/RLforHexapod/assets/urdf/robot.urdf
    tips = kin.tips(joint_pos)                      # (6, 3), 몸통 좌표, FOOT_ORDER 순서
    below = kin.feet_below_body(joint_pos, gravity) # (6,) = 단계 추정 신경망의 feet_below_body

joint_pos는 시뮬레이션 관절각 [rad] (robot_io.json의 action.joint_names 순서, 기본 자세 포함 절대값).
관절각을 재지 못하는 로봇에서는 마지막으로 보낸 목표각을 넣는다 (학습의 commanded_joints와 같음).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF = Path(__file__).resolve().parents[2] / "src" / "RLforHexapod" / "assets" / "urdf" / "robot.urdf"
# mdp/stair_reward.py FOOT_BODIES, mdp/observations.py TIP_OFFSET (tibia 바디 좌표의 발끝 추정점)
FOOT_ORDER = ["tibia_4", "tibia_5", "tibia_6", "tibia", "tibia_2", "tibia_3"]
TIP_OFFSET_L1 = (-0.004, -0.147, -0.021)
TIP_OFFSET = (0.004, 0.147, -0.021)


def rpy_matrix(roll, pitch, yaw) -> np.ndarray:
    """URDF rpy: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_matrix(axis, q) -> np.ndarray:
    """축 axis(단위 벡터) 둘레로 q만큼 돌리는 회전 (로드리게스)."""
    x, y, z = axis
    c, s, v = np.cos(q), np.sin(q), 1.0 - np.cos(q)
    return np.array([
        [c + x * x * v, x * y * v - z * s, x * z * v + y * s],
        [y * x * v + z * s, c + y * y * v, y * z * v - x * s],
        [z * x * v - y * s, z * y * v + x * s, c + z * z * v],
    ])


class HexapodKinematics:
    def __init__(self, urdf: str | Path = URDF, joint_names: list[str] | None = None):
        root = ET.parse(urdf).getroot()
        self.joints = {}  # child link → (joint name, type, parent link, origin R, origin p, axis)
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()]) if origin is not None else np.zeros(3)
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()] if origin is not None else [0, 0, 0]
            axis_el = j.find("axis")
            axis = np.array([float(v) for v in axis_el.get("xyz").split()]) if axis_el is not None else np.array([1.0, 0, 0])
            self.joints[j.find("child").get("link")] = (
                j.get("name"), j.get("type"), j.find("parent").get("link"), rpy_matrix(*rpy), xyz, axis / np.linalg.norm(axis))
        self.joint_names = joint_names  # None이면 dict로 받는다
        self.tip_offset = np.array([TIP_OFFSET_L1 if n == "tibia" else TIP_OFFSET for n in FOOT_ORDER])

    def _link_pose(self, link: str, q: dict) -> tuple[np.ndarray, np.ndarray]:
        """몸통 좌표에서 본 링크의 (R, p)."""
        if link not in self.joints:  # 뿌리 (body)
            return np.eye(3), np.zeros(3)
        name, jtype, parent, r0, p0, axis = self.joints[link]
        rp, pp = self._link_pose(parent, q)
        r = r0 @ axis_matrix(axis, q[name]) if jtype in ("revolute", "continuous") else r0
        return rp @ r, rp @ p0 + pp

    def _as_dict(self, joint_pos) -> dict:
        if isinstance(joint_pos, dict):
            return joint_pos
        if self.joint_names is None:
            raise ValueError("joint_names를 주지 않았으면 joint_pos는 {관절 이름: 각도} dict여야 합니다.")
        return dict(zip(self.joint_names, np.asarray(joint_pos, float), strict=True))

    def tips(self, joint_pos) -> np.ndarray:
        """발끝 추정점 (6, 3), 몸통 좌표, FOOT_ORDER 순서."""
        q = self._as_dict(joint_pos)
        out = np.zeros((len(FOOT_ORDER), 3))
        for i, link in enumerate(FOOT_ORDER):
            r, p = self._link_pose(link, q)
            out[i] = p + r @ self.tip_offset[i]
        return out

    def feet_below_body(self, joint_pos, projected_gravity) -> np.ndarray:
        """발끝 높이 - 몸통 원점 높이, 중력 방향 [m] (6,). projected_gravity = 몸통 좌표의 중력 단위 벡터."""
        up = -np.asarray(projected_gravity, float)
        up = up / np.linalg.norm(up)
        return self.tips(joint_pos) @ up
