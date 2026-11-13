"""L2 + Point-LIO에서 정책 입력을 만든다: 몸통 각속도, 중력 방향, 41×5 높이맵, 단계 신경망의 body_x.

준비 (로봇, ROS2):
    1. unitree_lidar_ros2 (unilidar_sdk2): L2 점군 /unilidar/cloud, IMU /unilidar/imu.
       작업 모드는 0 = 표준 FOV, 3D, IMU 켬(bit2 = 0), 이더넷, 전원 켜면 자동 시작.
       IMU를 끄는 비트(bit2 = 4)가 켜져 있으면 IMU 토픽이 오지 않는다. 그러면 이 노드가 경고한다.
    2. Point-LIO (point_lio_unilidar, L2 설정): gravity_align을 켠다. 자세 /aft_mapped_to_init, 월드 점군 /cloud_registered.
    3. base_in_imu: L2 IMU 좌표에서 본 몸통 원점(URDF base)의 위치·자세. CAD에서 읽는다.
       scripts/lidar_error/record_ros2.py의 --base_in_lio와 같은 값이다.

    python3 deploy/runtime/l2_sensors.py --base_in_imu x y z roll pitch yaw    # 확인용: 받은 값을 1초마다 출력

좌표 (heightmap.py와 같음): 지도 = Point-LIO 월드 (z 위), 몸통 = x 앞(라이다 쪽), y 왼쪽, z 위.
L2 IMU 축은 점군 좌표 축과 같은 방향이다 (Unitree: IMU 원점이 점군 좌표 (-0.0077, -0.0147, 0.0067) m, 회전 없음).
높이맵은 scripts/lidar_error/heightmap.py의 HeightMapPipeline으로 만든다 (측정 도구와 같은 코드).
"""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "lidar_error"))
from heightmap import HeightMapPipeline, Stairs, quat_to_matrix, rpy_to_matrix  # noqa: E402

BODY_X_CLIP = (-0.3, 0.6)   # mdp.lidar_body_x의 clip
NO_STAIRS_BODY_X = -3.0     # 수직면을 못 찾았을 때 (걷기 학습 env는 첫 수직면에서 약 -4.5 m에서 시작)
FALLBACK_HEIGHT = 0.24      # 높이맵 칸을 한 번도 못 본 데다 바닥도 모를 때 [m]


class L2Fusion:
    """ROS 없이 시험할 수 있는 계산 부분. 콜백이 on_imu / on_pose / on_cloud를 부르고, 정책 루프가 snapshot을 읽는다."""

    def __init__(self, base_in_imu, stairs: Stairs | None = None, tread: float = 0.3):
        x, y, z, roll, pitch, yaw = base_in_imu
        self.r_ib = rpy_to_matrix(roll, pitch, yaw)  # 몸통 좌표 → IMU 좌표
        self.p_ib = np.array([x, y, z], float)
        self.stairs = stairs or Stairs.uniform()
        self.tread = tread
        self.pipeline = HeightMapPipeline(self.stairs)
        self.lock = threading.Lock()
        self.gyro = np.zeros(3)
        self.gyro_t = 0.0
        self.r_wb = np.eye(3)
        self.p_wb = np.zeros(3)
        self.pose_t = 0.0
        self.height_map = np.full(41 * 5, np.nan)
        self.body_x = np.array([NO_STAIRS_BODY_X] + [BODY_X_CLIP[0]] * len(self.stairs.riser_x))
        self.map_t = 0.0
        self.floor_z = np.nan
        self.risers = np.full(len(self.stairs.riser_x), np.nan)
        self.lin_vel = np.zeros(3)

    # ----- 콜백 -----
    def on_imu(self, angular_velocity, t: float):
        """L2 IMU 각속도 (IMU 좌표) → 몸통 좌표."""
        w = self.r_ib.T @ np.asarray(angular_velocity, float)
        with self.lock:
            self.gyro, self.gyro_t = w, t

    def on_pose(self, position, quaternion_xyzw, t: float, linear_velocity=None):
        """Point-LIO 자세 (월드에서 본 IMU) → 월드에서 본 몸통. linear_velocity: Odometry twist (크기만 쓴다)."""
        r_wi = quat_to_matrix(quaternion_xyzw)
        r_wb = r_wi @ self.r_ib
        p_wb = np.asarray(position, float) + r_wi @ self.p_ib
        with self.lock:
            self.r_wb, self.p_wb, self.pose_t = r_wb, p_wb, t
            if linear_velocity is not None:
                self.lin_vel = np.asarray(linear_velocity, float)

    def on_cloud(self, points_world: np.ndarray, t: float):
        """Point-LIO 월드 점군 한 프레임 → 높이맵과 body_x."""
        with self.lock:
            r_wb, p_wb = self.r_wb.copy(), self.p_wb.copy()
        res = self.pipeline.step(points_world, p_wb, r_wb)
        hm = res.height_map.ravel()
        with self.lock:
            # 못 본 칸은 직전 값 (학습의 lidar_dropout처럼), 처음부터 못 본 칸은 평지로 본다
            fill = p_wb[2] - res.floor_z if np.isfinite(res.floor_z) else FALLBACK_HEIGHT
            old = np.where(np.isnan(self.height_map), fill, self.height_map)
            self.height_map = np.where(np.isnan(hm), old, hm)
            self.body_x = self._body_x(res.s_base, res.risers)
            self.floor_z = res.floor_z
            self.risers = res.risers
            self.map_t = t

    def _body_x(self, s_base: float, risers: np.ndarray) -> np.ndarray:
        """mdp.lidar_body_x와 같은 값: [첫 수직면 기준 x, 수직면마다 clip(x - 수직면)]. 못 찾은 수직면은 첫 수직면 + 단 깊이 × k."""
        if not np.isfinite(risers[0]):
            return np.array([NO_STAIRS_BODY_X] + [BODY_X_CLIP[0]] * len(risers))
        r = np.where(np.isfinite(risers), risers, risers[0] + self.tread * np.arange(len(risers)))
        return np.concatenate([[s_base - r[0]], np.clip(s_base - r, *BODY_X_CLIP)])

    def stairs_pose(self, x0: float) -> tuple[np.ndarray, np.ndarray]:
        """계단 좌표(첫 수직면 x = x0, 계단 가운데 선 y = 0, 바닥 z = 0)에서 본 몸통 (위치, 회전).
        규칙 단계 추정기(phase_rule.py)가 발끝을 이 좌표로 받는다. s축이 계단 가운데 선이라고 가정한다 (heightmap.py).
        수직면을 못 찾았으면 body_x와 같이 첫 수직면 NO_STAIRS_BODY_X 뒤의 평지로 본다
        (시뮬레이션 걷기 env는 계단 4 m 뒤에서 시작해 늘 단계 0이었다). 바닥을 모르면 몸통 높이 FALLBACK_HEIGHT."""
        with self.lock:
            loc = self.pipeline.locator
            height = self.p_wb[2] - self.floor_z if np.isfinite(self.floor_z) else FALLBACK_HEIGHT
            if loc is None:
                forward = self.r_wb[:, 0]
                yaw = np.arctan2(forward[1], forward[0])
                pos = np.array([x0 + NO_STAIRS_BODY_X, 0.0, height])
            else:
                rel = self.p_wb[:2] - loc.origin
                found = np.isfinite(self.risers[0])
                x = x0 + rel @ loc.dir - self.risers[0] if found else x0 + NO_STAIRS_BODY_X
                pos = np.array([x, rel @ loc.perp, height])
                yaw = np.arctan2(loc.dir[1], loc.dir[0])
            c, s = np.cos(-yaw), np.sin(-yaw)
            r_sw = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            return pos, r_sw @ self.r_wb

    def speed(self) -> float:
        """시뮬레이션 root_vel_w 노름과 같은 값: |(선속도, 각속도)|."""
        with self.lock:
            return float(np.sqrt(self.lin_vel @ self.lin_vel + self.gyro @ self.gyro))

    # ----- 정책 루프가 읽는 값 -----
    def snapshot(self) -> dict:
        with self.lock:
            gravity = self.r_wb.T @ np.array([0.0, 0.0, -1.0])
            return {
                "gyro": self.gyro.copy(), "gravity": gravity, "height_map": self.height_map.copy(),
                "body_x": self.body_x.copy(), "gyro_t": self.gyro_t, "pose_t": self.pose_t, "map_t": self.map_t,
                "base_pos": self.p_wb.copy(), "base_rot": self.r_wb.copy(),
            }


def main():
    import argparse

    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Imu, PointCloud2
    from sensor_msgs_py import point_cloud2

    ap = argparse.ArgumentParser(description="L2 + Point-LIO 센서 확인")
    ap.add_argument("--base_in_imu", type=float, nargs=6, required=True, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                    help="L2 IMU 좌표에서 본 몸통 원점 [m, 도]")
    ap.add_argument("--imu_topic", default="/unilidar/imu")
    ap.add_argument("--odom_topic", default="/aft_mapped_to_init")
    ap.add_argument("--cloud_topic", default="/cloud_registered")
    args = ap.parse_args()
    x, y, z, roll, pitch, yaw = args.base_in_imu
    fusion = L2Fusion((x, y, z, *np.radians([roll, pitch, yaw])))

    rclpy.init()
    node = Node("hexapod_l2_sensors")
    stamp = lambda msg: msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9  # noqa: E731
    node.create_subscription(Imu, args.imu_topic, lambda m: fusion.on_imu(
        (m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z), stamp(m)), qos_profile_sensor_data)
    node.create_subscription(Odometry, args.odom_topic, lambda m: fusion.on_pose(
        (m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z),
        (m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w),
        stamp(m), (m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z)), 10)
    node.create_subscription(PointCloud2, args.cloud_topic, lambda m: fusion.on_cloud(
        point_cloud2.read_points_numpy(m, field_names=("x", "y", "z")).astype(float), stamp(m)), 10)
    start = time.time()

    def report():
        s = fusion.snapshot()
        if s["gyro_t"] == 0.0 and time.time() - start > 2.0:
            node.get_logger().warn(f"{args.imu_topic}가 오지 않습니다. L2 작업 모드의 IMU 비트(bit2)가 0인지 확인하세요.")
        hm = s["height_map"].reshape(41, 5)
        node.get_logger().info(
            f"gyro {np.round(s['gyro'], 3)} | gravity {np.round(s['gravity'], 3)} | body_x {np.round(s['body_x'], 3)} | "
            f"높이맵 가운데 줄 앞/몸통 아래 {hm[-1, 2]:.3f}/{hm[12, 2]:.3f} m | 바닥 z {fusion.floor_z:.3f}")

    node.create_timer(1.0, report)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
