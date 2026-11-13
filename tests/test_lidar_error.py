"""라이다 오차 측정 도구 (scripts/lidar_error) 시험: 합성 데이터에 넣은 오차를 되찾는지 본다.
torch·Isaac Lab 없이 돈다."""

import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "lidar_error"))
from analyze import analyze_static, analyze_walk, summarize  # noqa: E402
from heightmap import GX, HeightMapPipeline, Stairs, load_trial, truth_height_map  # noqa: E402
from synth import direct_trial, make_meta, point_frames  # noqa: E402


def _as_trial(log):
    return {"meta": log.meta, **{k: np.asarray(v) for k, v in log.rows.items()}}


def test_stairs_height_matches_sim_definition():
    s = Stairs.uniform(0.16, 0.3, 3, 0.8, 1.0)
    x = np.array([-0.1, 0.0, 0.29, 0.31, 0.65, 1.39, 1.41])
    assert np.allclose(s.height(x, np.zeros_like(x)), [0, 0.16, 0.16, 0.32, 0.48, 0.48, 0])
    assert s.height(0.1, 0.6) == 0.0  # 폭 밖


@pytest.mark.parametrize("true_x", [-0.25, -0.1, 0.12])
def test_pipeline_finds_risers_and_height_map(true_x):
    rng = np.random.default_rng(1)
    stairs = Stairs.uniform()
    meta = make_meta("p", "static", true_x, stairs=stairs)
    pipe = HeightMapPipeline(stairs)
    for pts, pos, rot in point_frames(meta, rng, frames=15):
        r = pipe.step(pts, pos, rot)
    assert abs(r.body_x - true_x) < 0.01
    assert np.all(np.isfinite(r.risers))
    assert np.allclose(np.diff(r.risers), 0.3, atol=0.01)
    truth = truth_height_map(stairs, true_x, 0.0, meta["true_z"])
    far = np.min(np.abs((true_x + GX)[..., None] - stairs.edges), axis=-1) > 0.02
    err = (r.height_map - truth)[far]
    assert np.isfinite(err).all()
    assert np.median(np.abs(err)) < 0.003


def test_pipeline_pose_jitter_moves_body_x_and_map_together():
    """Point-LIO 자세 흔들림은 몸통 x와 높이맵을 같이 민다 (accumulated 가정)."""
    rng = np.random.default_rng(2)
    stairs = Stairs.uniform()
    meta = make_meta("j", "static", -0.1, stairs=stairs)
    pipe = HeightMapPipeline(stairs)
    from heightmap import TrialLog
    log = TrialLog(meta)
    for f, (pts, pos, rot) in enumerate(point_frames(meta, rng, frames=60, pose_noise=0.005)):
        log.add(pipe.step(pts, pos, rot), f * 0.1, f * 0.1, len(pts))
    res = analyze_static(_as_trial(log), warmup=1.0)
    assert 0.002 < res["x_noise"] < 0.008
    assert abs(res["x_bias"]) < 0.008


def test_static_analysis_recovers_injected_errors():
    rng = np.random.default_rng(3)
    meta = make_meta("s", "static", -0.12)
    inj = dict(x_bias=0.012, x_noise=0.004, range_bias=-0.007, pitch=math.radians(0.8), roll=math.radians(-0.5),
               height_noise=0.004, dropout=0.1, period=0.1)
    res = analyze_static(_as_trial(direct_trial(meta, rng, **inj)), warmup=0.0)
    assert res["x_bias"] == pytest.approx(0.012, abs=0.001)
    assert res["x_noise"] == pytest.approx(0.004, abs=0.0008)
    assert res["shift_mean"] == pytest.approx(0.012, abs=0.01)  # 2 cm 격자라 ±1 cm
    assert res["range_bias"] == pytest.approx(-0.007, abs=0.001)
    assert res["pitch"] == pytest.approx(math.radians(0.8), abs=math.radians(0.15))
    assert res["roll"] == pytest.approx(math.radians(-0.5), abs=math.radians(0.15))
    assert res["height_noise"] == pytest.approx(0.004, abs=0.0008)
    assert res["dropout"] == pytest.approx(0.1, abs=0.02)
    assert res["period"] == pytest.approx(0.1)


def test_walk_drift_and_summary(tmp_path):
    rng = np.random.default_rng(4)
    walk = analyze_walk(_as_trial(direct_trial(make_meta("w", "walk", -0.4, true_x_end=0.9), rng,
                                               x_bias=0.005, x_noise=0.002, drift=0.026, duration=40.0)), still=2.0)
    assert walk["drift"] == pytest.approx(0.026, abs=0.002)
    assert walk["drift_per_m"] == pytest.approx(0.02, abs=0.002)

    statics = []
    for k, x in enumerate((-0.2, -0.1, 0.0, 0.1)):
        log = direct_trial(make_meta(f"s{k}", "static", x), rng, x_bias=0.015 * (-1) ** k, x_noise=0.003)
        path = tmp_path / f"s{k}.npz"
        log.save(path)
        statics.append(analyze_static(load_trial(path), warmup=0.0))
    summary = summarize(statics + [walk], margin=1.5, climb_length=1.0)
    assert summary["map_mode"]["verdict"].startswith("accumulated")
    assert summary["measured"]["lidar_x_bias"] == pytest.approx(0.015 + 0.02, abs=0.003)
    assert summary["suggested"]["lidar_x_noise"] == pytest.approx(1.5 * summary["measured"]["lidar_x_noise"])
    assert "lidar_x_bias" in summary["current"]  # train_phase_net.py TRAIN_ERRORS를 읽었다


def test_per_frame_map_is_detected():
    rng = np.random.default_rng(5)
    statics = [analyze_static(_as_trial(direct_trial(make_meta(f"s{k}", "static", -0.1), rng,
                                                     x_bias=0.015 * (-1) ** k, map_follows_x=False)), 0.0)
               for k in range(4)]
    assert summarize(statics, 1.5, 1.0)["map_mode"]["verdict"] == "per_frame"
