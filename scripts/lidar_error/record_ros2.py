"""라이다 오차 측정 기록 노드 (ROS2). Point-LIO 출력을 받아 policy가 볼 높이맵과 몸통 x를 trial 파일로 저장한다.

Point-LIO가 돌고 있어야 한다 (실시간, 또는 ros2 bag play로 다시 돌려도 된다). gravity_align을 켠다.
    입력  --odom_topic  nav_msgs/Odometry      Point-LIO 몸통(IMU) 자세, 월드 좌표 (기본 /aft_mapped_to_init)
          --cloud_topic sensor_msgs/PointCloud2 월드 좌표로 옮긴 이번 스캔 (기본 /cloud_registered)
    출력  <out>/<name>.npz (heightmap.TrialLog 형식)

예 (정지, 몸통 원점이 첫 수직면 0.20 m 앞, 바닥에서 0.17 m):
    python3 record_ros2.py --name s_m020_r01 --kind static --true_x -0.20 --true_z 0.17 --duration 30
예 (걷기, -0.40에서 출발해 0.95에서 멈춤):
    python3 record_ros2.py --name w_r01 --kind walk --true_x -0.40 --true_x_end 0.95 --true_z 0.17 --duration 60

trial마다 Point-LIO를 새로 켠다: 켤 때마다 달라지는 편향(lidar_x_bias, 에피소드 고정 오차)을 재기 위해서다.
--base_in_lio는 Point-LIO 몸통 좌표에서 본 로봇 몸통 원점(URDF base)의 자세다. 틀리면 그만큼 오차로 잡힌다.
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heightmap import HeightMapPipeline, Stairs, TrialLog, quat_to_matrix, rpy_to_matrix  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True, help="trial 이름 (파일 이름)")
    p.add_argument("--kind", choices=("static", "walk"), required=True)
    p.add_argument("--true_x", type=float, required=True, help="첫 수직면 기준 몸통 원점 x [m] (계단 앞이 음수)")
    p.add_argument("--true_z", type=float, required=True, help="바닥 기준 몸통 원점 높이 [m]")
    p.add_argument("--true_y", type=float, default=0.0, help="계단 가운데 선 기준 몸통 y [m] (왼쪽 +)")
    p.add_argument("--true_yaw_deg", type=float, default=0.0, help="계단 x축 기준 몸통 heading [deg]")
    p.add_argument("--true_x_end", type=float, default=None, help="walk: 끝에 멈춘 위치의 true_x [m]")
    p.add_argument("--stairs", default=None, help="실측 계단 형상 json (heightmap.Stairs.to_dict 형식)")
    p.add_argument("--step_height", type=float, default=0.16, help="--stairs가 없을 때 고른 계단")
    p.add_argument("--tread", type=float, default=0.3)
    p.add_argument("--num_steps", type=int, default=3)
    p.add_argument("--top_length", type=float, default=0.8)
    p.add_argument("--width", type=float, default=1.0)
    p.add_argument("--base_in_lio", type=float, nargs=6, default=None, metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
                   help="Point-LIO 몸통 좌표에서 본 로봇 몸통 원점 [m, deg]. 필수에 가깝다")
    p.add_argument("--odom_topic", default="/aft_mapped_to_init")
    p.add_argument("--cloud_topic", default="/cloud_registered")
    p.add_argument("--duration", type=float, default=30.0, help="기록 시간 [s] (첫 프레임부터). 0 = Ctrl+C까지")
    p.add_argument("--out", default="lidar_trials")
    p.add_argument("--note", default="")
    return p.parse_args()


def main():
    args = parse_args()
    import rclpy
    from nav_msgs.msg import Odometry
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs_py import point_cloud2

    if args.stairs:
        with open(args.stairs) as f:
            stairs = Stairs.from_dict(json.load(f))
    else:
        stairs = Stairs.uniform(args.step_height, args.tread, args.num_steps, args.top_length, args.width)
    if args.kind == "walk" and args.true_x_end is None:
        sys.exit("walk는 --true_x_end가 필요하다")
    if args.base_in_lio is None:
        print("[경고] --base_in_lio가 없어 Point-LIO 몸통 = 로봇 몸통 원점으로 둔다. 장착 위치만큼 오차로 잡힌다")
        args.base_in_lio = [0.0] * 6
    t_lb = np.array(args.base_in_lio[:3])
    r_lb = rpy_to_matrix(*np.radians(args.base_in_lio[3:]))

    meta = {
        "name": args.name, "kind": args.kind, "true_x": args.true_x, "true_y": args.true_y, "true_z": args.true_z,
        "true_yaw_deg": args.true_yaw_deg, "true_x_end": args.true_x_end, "stairs": stairs.to_dict(),
        "base_in_lio": args.base_in_lio, "odom_topic": args.odom_topic, "cloud_topic": args.cloud_topic,
        "note": args.note, "recorded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    log = TrialLog(meta)
    pipeline = HeightMapPipeline(stairs)

    class Recorder(Node):
        def __init__(self):
            super().__init__("lidar_error_recorder")
            self.odoms = deque(maxlen=200)  # (stamp, 위치, 회전)
            self.t0 = None
            self.done = False
            self.create_subscription(Odometry, args.odom_topic, self.on_odom, qos_profile_sensor_data)
            self.create_subscription(PointCloud2, args.cloud_topic, self.on_cloud, qos_profile_sensor_data)

        def on_odom(self, msg):
            p, q = msg.pose.pose.position, msg.pose.pose.orientation
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.odoms.append((stamp, np.array([p.x, p.y, p.z]), quat_to_matrix((q.x, q.y, q.z, q.w))))

        def on_cloud(self, msg):
            if self.done or not self.odoms:
                return
            t_recv = time.time()
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            o_stamp, pos, rot = min(self.odoms, key=lambda o: abs(o[0] - stamp))
            if abs(o_stamp - stamp) > 0.05:
                self.get_logger().warn(f"스캔과 자세 시각 차이 {abs(o_stamp - stamp):.3f}s → 건너뜀")
                return
            pts = point_cloud2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True).astype(float)
            base_pos, base_rot = pos + rot @ t_lb, rot @ r_lb
            r = pipeline.step(pts, base_pos, base_rot)
            log.add(r, stamp, t_recv, len(pts))
            if self.t0 is None:
                self.t0 = stamp
            n = len(log)
            if n % 10 == 0:
                err = r.body_x - args.true_x if args.kind == "static" else float("nan")
                seen = np.mean(~np.isnan(r.height_map))
                self.get_logger().info(f"{n} 프레임, body_x {r.body_x:+.3f} (static 오차 {err:+.3f}), "
                                       f"높이맵 채움 {seen:.0%}, 점 {len(pts)}")
            if args.duration > 0 and stamp - self.t0 >= args.duration:
                self.done = True

    rclpy.init()
    node = Recorder()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        os.makedirs(args.out, exist_ok=True)
        path = os.path.join(args.out, f"{args.name}.npz")
        if len(log):
            log.save(path)
            print(f"저장: {path} ({len(log)} 프레임)")
        else:
            print("받은 프레임이 없어 저장하지 않았다. 토픽 이름과 Point-LIO가 도는지 확인할 것")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
