"""합성 trial: 오차를 알고 넣은 가짜 측정. analyze.py가 넣은 값을 되찾는지 확인하고, 보고서 모양을 미리 본다.

    python3 scripts/lidar_error/synth.py --out /tmp/lidar_synth
    python3 scripts/lidar_error/analyze.py /tmp/lidar_synth

두 가지를 만든다.
    direct_trial  높이맵·몸통 x에 오차를 바로 넣는다 (분석 계산 확인용, 빠름)
    point_frames  계단 표면 점군을 만들어 지도 좌표로 옮긴다 (HeightMapPipeline 확인용)
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heightmap import GX, GY, FrameResult, Stairs, TrialLog, rpy_to_matrix, truth_height_map  # noqa: E402


def make_meta(name, kind, true_x, true_z=0.17, stairs=None, true_x_end=None, true_y=0.0, true_yaw_deg=0.0):
    return {"name": name, "kind": kind, "true_x": true_x, "true_y": true_y, "true_z": true_z,
            "true_yaw_deg": true_yaw_deg, "true_x_end": true_x_end, "stairs": (stairs or Stairs()).to_dict(),
            "note": "synthetic"}


def direct_trial(meta, rng, x_bias=0.0, x_noise=0.0, range_bias=0.0, pitch=0.0, roll=0.0, height_noise=0.0,
                 dropout=0.0, period=0.1, duration=20.0, map_follows_x=True, drift=0.0) -> TrialLog:
    """오차를 높이맵·몸통 x에 바로 넣는다. sensor_model._distort와 같은 식.
    map_follows_x: 높이맵이 몸통 x 오차만큼 밀린다 (accumulated).
    walk면 true_x → true_x_end로 움직이고 drift [m]가 쌓인다."""
    stairs = Stairs.from_dict(meta["stairs"])
    log = TrialLog(meta)
    n = int(duration / period)
    walk = meta["kind"] == "walk"
    last = None
    for f in range(n):
        u = f / (n - 1)
        if walk:  # 앞 20 %, 뒤 20 %는 정지
            a = np.clip((u - 0.2) / 0.6, 0.0, 1.0)
            x = meta["true_x"] + a * (meta["true_x_end"] - meta["true_x"])
            err = x_bias + a * drift + rng.normal(0.0, x_noise)
        else:
            x = meta["true_x"]
            err = x_bias + rng.normal(0.0, x_noise)
        shift = err if map_follows_x else 0.0
        hm = truth_height_map(stairs, x + shift, meta["true_y"], meta["true_z"], math.radians(meta["true_yaw_deg"]))
        hm = hm + range_bias - pitch * GX + roll * GY + rng.normal(0.0, height_noise, hm.shape)
        fresh = rng.random(hm.shape) >= dropout
        if last is not None:
            hm = np.where(fresh, hm, last)
        else:
            fresh[:] = True
        last = hm
        r = FrameResult(np.zeros(6), x, np.array([0.0]), 0.0, x + err, hm, fresh)
        log.add(r, f * period, f * period + 0.03, 0)
    return log


def surface_points(stairs: Stairs, base_x, rng, n=8000, noise=0.003, view=(-0.4, 1.6), half_width=0.6):
    """계단 좌표의 표면 점 (N, 3): 바닥과 윗면 + 몸통 앞쪽 수직면. 가려짐은 무시 (윗면은 위에서 다 보인다)."""
    x = rng.uniform(base_x + view[0], base_x + view[1], n)
    y = rng.uniform(-half_width, half_width, n)
    pts = [np.stack([x, y, stairs.height(x, y)], axis=1)]
    prev = 0.0
    for rx, top in zip(stairs.riser_x, stairs.top_z, strict=True):
        if base_x + view[0] < rx < base_x + view[1]:
            m = n // 20
            z = rng.uniform(prev, top, m)
            yy = rng.uniform(-stairs.width / 2, stairs.width / 2, m)
            pts.append(np.stack([np.full(m, rx), yy, z], axis=1))
        prev = top
    pts = np.concatenate(pts)
    return pts + rng.normal(0.0, noise, pts.shape)


def point_frames(meta, rng, frames=30, map_yaw=0.7, map_offset=(1.3, -0.4, 0.25), pose_noise=0.0, **kw):
    """지도 좌표 (계단 좌표를 map_yaw만큼 돌리고 map_offset만큼 옮김)의 점군과 몸통 자세를 프레임마다 낸다.
    pose_noise: 프레임마다 몸통·점군이 같이 밀리는 x 흔들림 std [m] (Point-LIO 자세 흔들림)."""
    stairs = Stairs.from_dict(meta["stairs"])
    r_ms = rpy_to_matrix(0.0, 0.0, map_yaw)
    t_ms = np.asarray(map_offset, float)
    base_st = np.array([meta["true_x"], meta["true_y"], meta["true_z"]])
    r_base = r_ms @ rpy_to_matrix(0.0, 0.0, math.radians(meta["true_yaw_deg"]))
    for _ in range(frames):
        jitter = np.array([rng.normal(0.0, pose_noise), 0.0, 0.0])
        pts = surface_points(stairs, meta["true_x"], rng, **kw) + jitter
        yield (pts @ r_ms.T + t_ms, r_ms @ (base_st + jitter) + t_ms, r_base)


def demo_set(out, seed=0):
    """보고서 확인용: 지금 학습 오차 수준의 static 20개 + walk 3개."""
    rng = np.random.default_rng(seed)
    os.makedirs(out, exist_ok=True)
    for x in (-0.30, -0.15, 0.0, 0.15, 0.30):
        for k in range(4):
            meta = make_meta(f"s_{x:+.2f}_r{k}", "static", x)
            direct_trial(meta, rng, x_bias=rng.uniform(-0.012, 0.012), x_noise=0.004,
                         range_bias=rng.uniform(-0.006, 0.006), pitch=math.radians(rng.uniform(-0.6, 0.6)),
                         roll=math.radians(rng.uniform(-0.6, 0.6)), height_noise=0.004, dropout=0.03
                         ).save(os.path.join(out, meta["name"] + ".npz"))
    for k in range(3):
        meta = make_meta(f"w_r{k}", "walk", -0.40, true_x_end=0.95)
        direct_trial(meta, rng, x_bias=rng.uniform(-0.01, 0.01), x_noise=0.004, drift=rng.uniform(-0.02, 0.02),
                     duration=40.0).save(os.path.join(out, meta["name"] + ".npz"))
    print(f"합성 trial 23개: {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    demo_set(args.out, args.seed)


if __name__ == "__main__":
    main()
