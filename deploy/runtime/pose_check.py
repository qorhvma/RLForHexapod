"""1단계 실물 검사: sim 관절각 → 실물 관절 각도 변환(joint_map)이 맞는지 눈으로 확인한다.

★ 로봇을 받침대에 올려 다리 6개가 모두 공중에 뜬 상태에서만 돌린다. 전원 스위치에 손을 둔다.

    python3 deploy/runtime/pose_check.py --hexacore ~/HexaCore            # 로봇 Pi (ESP32 = /dev/ttyAMA0)
    python3 deploy/runtime/pose_check.py --hexacore ../../HexaCore-clone --sim   # 서보 없이 계산만

하는 일
    1. HexaCore 서기 자세(실제 각도 180/135/45)로 보낸 뒤, 3초 동안 천천히 sim 기본 자세로 옮긴다.
       sim 기본 자세 = femur 20°, tibia -110° (실제 각도 hip 180, femur 250, tibia 158.4):
       허벅지가 수평에서 70° 아래로 내려가고, 무릎은 조금(21.6°) 접혀 종아리가 거의 수직으로 선다. 키가 큰 자세다.
    2. 'all'을 입력하면 펌웨어 다리 0~5 × hip/femur/tibia를 차례로 sim 각도 +step만큼 움직였다가 돌아온다.
       화면에 "보여야 하는 움직임"이 나온다. 맞으면 y, 다르면 n과 본 것을 적는다. 결과는 --log 파일에 쌓인다.
       이것으로 세 가지를 한 번에 확인한다: 다리 번호(어느 다리가 움직이나), 관절(어느 마디), 방향(어느 쪽으로).

명령: all | <다리 0~5> <h|f|t> <+|-> | d (sim 기본 자세) | s (HexaCore 서기 자세) | q (서기 자세 → 힘 빼고 끝)
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import joint_map  # noqa: E402

JOINT_KEYS = {"h": 0, "f": 1, "t": 2}
JOINT_LABELS = ("hip", "femur", "tibia")
RATE_HZ = 50


class Robot:
    def __init__(self, args):
        sys.path.insert(0, str(Path(args.hexacore).expanduser() / "software" / "raspberrypi"))
        import firmware_link  # noqa: PLC0415
        import leg_model  # noqa: PLC0415

        self.lm = leg_model
        self.fw = firmware_link.FirmwareLink(args.port, simulate=True if args.sim else None)
        self.limits = np.array(leg_model.JOINT_LIMITS)  # (6, 3, 2) 실제 각도 허용 범위 (펌웨어 틱 한계 포함)
        self.pose = None  # 마지막으로 보낸 실제 각도 (6, 3)

    def start(self):
        self.fw.connect()
        if self.fw.simulate:
            print("[pose_check] 시리얼이 없어 시뮬레이션 모드 (서보로 아무것도 안 나감)")

    def out_of_limits(self, true):
        bad = []
        for leg in range(6):
            for j in range(3):
                lo, hi = self.limits[leg, j]
                if not lo <= true[leg, j] <= hi:
                    bad.append(f"다리 {leg} {JOINT_LABELS[j]} {true[leg, j]:.1f}° (허용 {lo:.1f}~{hi:.1f})")
        return bad

    def send(self, true, full=False):
        fw = self.lm.pose_to_firmware(true.tolist())
        self.fw.send_pose(fw, full=full)
        self.pose = true.copy()

    def move(self, target, seconds):
        """실제 각도 공간에서 직선으로 seconds 동안 옮긴다. 허용 범위 밖이면 보내지 않는다."""
        bad = self.out_of_limits(target)
        if bad:
            print("[pose_check] 허용 범위 밖이라 보내지 않음:\n   " + "\n   ".join(bad))
            return False
        if self.pose is None:
            self.send(target, full=True)
            return True
        start, n = self.pose.copy(), max(1, int(seconds * RATE_HZ))
        t0 = time.monotonic()
        for i in range(1, n + 1):
            self.send(start + (target - start) * i / n)
            time.sleep(max(0.0, t0 + i / RATE_HZ - time.monotonic()))
        self.send(target, full=True)
        return True

    def stop(self):
        self.fw.close()  # 힘 빼기(o) 후 종료


def print_pose(robot, true, title):
    fw = robot.lm.pose_to_firmware(true.tolist())
    print(f"\n{title}  (실제 각도 → 펌웨어 각도)")
    for leg in range(6):
        cells = "  ".join(f"{JOINT_LABELS[j]} {true[leg, j]:6.1f} → {fw[leg][j]:6.1f}" for j in range(3))
        print(f"  다리 {leg} {joint_map.FW_NAMES[leg]:<5} ({joint_map.FW_TO_URDF[leg]})  {cells}")


def nudge(default_true, leg, joint, sign, step):
    """sim 각도로 ±step 움직인 실제 각도."""
    t = default_true.copy()
    side = joint_map.SIDE[leg]
    delta = step * sign
    t[leg, joint] += (side * delta, -delta, delta)[joint]
    return t


def ask_one(robot, default_true, leg, joint, sign, step, log):
    target = nudge(default_true, leg, joint, sign, step)
    expect = joint_map.describe(leg, joint, sign)
    print(f"\n다리 {leg} ({joint_map.FW_NAMES[leg]}, {joint_map.FW_TO_URDF[leg]}) {JOINT_LABELS[joint]} sim {'+' if sign > 0 else '-'}{step:g}°"
          f"  실제 {default_true[leg, joint]:.1f} → {target[leg, joint]:.1f}°")
    print(f"  보여야 하는 움직임: {joint_map.FW_NAMES[leg]} 다리 — {expect}")
    if not robot.move(target, 1.0):
        return
    ans = input("  맞나요? [y / n 본 것 / 엔터 = 건너뜀] ").strip()
    robot.move(default_true, 1.0)
    if ans:
        rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "fw_leg": leg, "urdf_leg": joint_map.FW_TO_URDF[leg],
               "joint": JOINT_LABELS[joint], "sim_delta_deg": step * sign, "expect": expect,
               "ok": ans.lower().startswith("y"), "note": ans}
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hexacore", default="~/HexaCore", help="HexaCore 저장소 (leg_model, firmware_link, calibration.json)")
    ap.add_argument("--port", default="/dev/ttyAMA0", help="ESP32 시리얼 (로봇 Pi는 GPIO UART)")
    ap.add_argument("--sim", action="store_true", help="서보 없이 계산만")
    ap.add_argument("--io", default=str(HERE.parent / "mixed_phase_v1" / "robot_io.json"))
    ap.add_argument("--step", type=float, default=10.0, help="움직여 볼 sim 각도 [도]")
    ap.add_argument("--log", default="pose_check_log.jsonl")
    args = ap.parse_args()

    io = json.loads(Path(args.io).read_text(encoding="utf-8"))["action"]
    default_true = joint_map.sim_to_true(io["default_joint_pos_rad"], io["joint_names"])
    robot = Robot(args)
    stand_true = np.array(robot.lm.STAND_POSE, float)
    print_pose(robot, stand_true, "HexaCore 서기 자세")
    print_pose(robot, default_true, "sim 기본 자세")
    bad = robot.out_of_limits(default_true)
    if bad:
        print("\n[pose_check] sim 기본 자세가 HexaCore 허용 범위 밖이라 멈춥니다:\n   " + "\n   ".join(bad))
        return 1

    print("\n★ 로봇이 받침대 위에 있고 다리 6개가 모두 공중에 떠 있나요? 전원 스위치에 손을 두세요.")
    if input("  준비됐으면 yes 입력: ").strip().lower() != "yes":
        return 0
    robot.start()
    try:
        robot.move(stand_true, 0)  # 처음 한 번은 서보가 지금 위치를 모르니 바로 간다
        time.sleep(1.5)
        print("\n서기 자세 → sim 기본 자세 (3초)")
        robot.move(default_true, 3.0)
        while True:
            cmd = input("\n명령 [all | 다리 h/f/t +/- | d | s | q]: ").strip().split()
            if not cmd:
                continue
            if cmd[0] == "q":
                break
            if cmd[0] == "d":
                robot.move(default_true, 2.0)
            elif cmd[0] == "s":
                robot.move(stand_true, 2.0)
            elif cmd[0] == "all":
                for leg in range(6):
                    for joint in range(3):
                        ask_one(robot, default_true, leg, joint, +1, args.step, args.log)
                print(f"\n끝. 결과: {args.log}")
            elif len(cmd) == 3 and cmd[0].isdigit() and cmd[1] in JOINT_KEYS and cmd[2] in "+-":
                ask_one(robot, default_true, int(cmd[0]), JOINT_KEYS[cmd[1]], 1 if cmd[2] == "+" else -1, args.step, args.log)
            else:
                print("  예) all   0 f +   3 h -   d   s   q")
        print("서기 자세로 돌아간 뒤 힘을 뺍니다")
        robot.move(stand_true, 3.0)
    finally:
        robot.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
