"""라이다 오차 분석: record_ros2.py가 남긴 trial 파일들을 참값과 비교해 sensor_model.SensorErrorCfg 값을 낸다.

    python3 scripts/lidar_error/analyze.py lidar_trials/ --json lidar_error_summary.json

static trial (정지, 참 자세를 줄자로 잼)
    몸통 x 오차 e_x = body_x - true_x       → 평균 = 이번에 켠 편향 (lidar_x_bias), 표준편차 = lidar_x_noise
    높이맵 오차 e = 높이맵 - 참 높이맵      → 프레임마다 아래 순서로 나눈다
        1) 지도 밀림 s: 참 높이맵을 x로 밀어 가며 오차가 가장 작은 s (모서리가 격자 안에 있을 때만)
        2) 모서리 근처(2 cm)를 뺀 칸에 e = b - pitch·x + roll·y + r 최소제곱 (sensor_model._distort와 같은 부호)
           b → lidar_range_bias, (roll, pitch) → lidar_mount_tilt, r의 표준편차 → 높이맵 GaussianNoiseCfg std
        3) 칸이 이번 프레임에 갱신되지 않은 비율 → lidar_dropout, 프레임 간격 → lidar_period
    map_mode 확인: trial별 지도 밀림 s와 e_x 평균이 같으면 accumulated, s ≈ 0이면 per_frame
walk trial (시작·끝에 정지)
    끝의 e_x - 시작의 e_x = 걷는 동안 쌓인 x 드리프트 → m당 드리프트 × climb_length를 lidar_x_bias에 더한다

학습 최댓값 제안 = 측정값(trial들의 95%) × margin. 지금 학습 값(train_phase_net.py TRAIN_ERRORS)과 비교해 보여 준다.
"""

import argparse
import ast
import glob
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heightmap import GRID_X, GX, GY, Stairs, load_trial, truth_height_map  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SHIFTS = np.arange(-0.10, 0.10 + 1e-9, 0.001)  # 지도 밀림 후보 [m]
EDGE_MARGIN = 0.02  # 모서리에서 이 거리 안의 칸은 기울기·노이즈 적합에서 뺀다 [m]
MIN_FIT_CELLS = 20


def _finite_mean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def _finite_std(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if len(a) > 1 else float("nan")


def _lag1(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    a = a - a.mean() if len(a) else a
    if len(a) < 3 or a.var() < 1e-18:
        return float("nan")
    return float((a[1:] * a[:-1]).mean() / a.var())


def analyze_static(trial: dict, warmup: float) -> dict:
    meta = trial["meta"]
    stairs = Stairs.from_dict(meta["stairs"])
    tx, ty, tz, yaw = meta["true_x"], meta["true_y"], meta["true_z"], math.radians(meta["true_yaw_deg"])
    t = trial["t_stamp"] - trial["t_stamp"][0]
    keep = t >= warmup
    body_x = trial["body_x"][keep]
    hm = trial["height_map"][keep].astype(float)
    fresh = trial["fresh"][keep].astype(bool)
    x_err = body_x - tx

    # 1) 지도 밀림
    edges = stairs.edges
    observable = bool(np.any((edges > tx + GRID_X[0] + 0.03) & (edges < tx + GRID_X[-1] - 0.03)))
    shift = np.full(len(hm), np.nan)
    if observable and len(hm):
        truth_c = np.stack([truth_height_map(stairs, tx + s, ty, tz, yaw) for s in SHIFTS])  # (C, 41, 5)
        e = hm[None] - truth_c[:, None]  # (C, F, 41, 5)
        with np.errstate(invalid="ignore"):
            centered = e - np.nanmedian(e, axis=(2, 3), keepdims=True)
            cost = np.nanmean(np.abs(centered), axis=(2, 3))  # (C, F)
        # 격자가 2 cm라 모서리가 칸 사이를 지날 때만 비용이 바뀐다
        # → 최소가 되는 구간의 가운데를 쓴다 (±1 cm까지만 보인다)
        ok = np.isfinite(cost).all(axis=0)
        best = np.isclose(cost[:, ok], cost[:, ok].min(axis=0), rtol=0.0, atol=1e-12)
        shift[ok] = (SHIFTS[:, None] * best).sum(axis=0) / best.sum(axis=0)

    # 2) 높이 편향·기울기·노이즈
    b, pitch, roll = (np.full(len(hm), np.nan) for _ in range(3))
    residuals, edge_miss, edge_cells = [], 0, 0
    half_step = 0.5 * min(np.diff([0.0, *stairs.top_z]).min(), stairs.top_z[-1])
    for f in range(len(hm)):
        s = shift[f] if np.isfinite(shift[f]) else 0.0
        e = hm[f] - truth_height_map(stairs, tx + s, ty, tz, yaw)
        x_st = tx + s + GX  # 칸의 계단 x (heading은 작다고 본다)
        near_edge = np.min(np.abs(x_st[..., None] - edges), axis=-1) < EDGE_MARGIN
        valid = np.isfinite(e)
        edge_cells += int((valid & near_edge).sum())
        edge_miss += int((valid & near_edge & (np.abs(e) > half_step)).sum())
        fit = valid & ~near_edge
        if fit.sum() < MIN_FIT_CELLS:
            continue
        a = np.stack([np.ones(fit.sum()), -GX[fit], GY[fit]], axis=1)
        coef, *_ = np.linalg.lstsq(a, e[fit], rcond=None)
        b[f], pitch[f], roll[f] = coef
        residuals.append(e[fit] - a @ coef)
    r = np.concatenate(residuals) if residuals else np.array([])

    # 3) 구멍·주기
    seen = np.isfinite(hm)
    stale = seen & ~fresh
    dt = np.diff(trial["t_stamp"])
    latency = trial["t_recv"] - trial["t_stamp"]

    return {
        "name": meta["name"], "kind": "static", "true_x": tx, "frames": int(len(hm)),
        "lost": float(np.mean(~np.isfinite(body_x))) if len(body_x) else float("nan"),
        "x_bias": _finite_mean(x_err), "x_noise": _finite_std(x_err), "x_rho1": _lag1(x_err),
        "shift_observable": observable, "shift_mean": _finite_mean(shift), "shift_std": _finite_std(shift),
        "range_bias": _finite_mean(b), "pitch": _finite_mean(pitch), "roll": _finite_mean(roll),
        "tilt_std": float(np.nanstd(np.concatenate([pitch, roll]))) if np.isfinite(pitch).any() else float("nan"),
        "height_noise": float(r.std()) if len(r) > 1 else float("nan"),
        "height_noise_robust": float(1.4826 * np.median(np.abs(r - np.median(r)))) if len(r) else float("nan"),
        "edge_miss": edge_miss / edge_cells if edge_cells else float("nan"),
        "dropout": float(stale.sum() / seen.sum()) if seen.any() else float("nan"),
        "unseen": float(1.0 - seen.mean()) if seen.size else float("nan"),
        "period": float(np.median(dt)) if len(dt) else float("nan"),
        "period_p95": float(np.percentile(dt, 95)) if len(dt) else float("nan"),
        "latency": float(np.median(latency)),
    }


def analyze_walk(trial: dict, still: float) -> dict:
    meta = trial["meta"]
    t = trial["t_stamp"] - trial["t_stamp"][0]
    body_x = trial["body_x"]
    start = _finite_mean(body_x[t <= still]) - meta["true_x"]
    end = _finite_mean(body_x[t >= t[-1] - still]) - meta["true_x_end"]
    dist = abs(meta["true_x_end"] - meta["true_x"])
    return {
        "name": meta["name"], "kind": "walk", "true_x": meta["true_x"], "true_x_end": meta["true_x_end"],
        "frames": int(len(t)), "lost": float(np.mean(~np.isfinite(body_x))),
        "x_err_start": start, "x_err_end": end, "drift": end - start,
        "drift_per_m": (end - start) / dist if dist > 0 else float("nan"),
    }


def _p95_abs(values):
    v = np.abs(np.asarray([x for x in values if np.isfinite(x)], float))
    return float(np.percentile(v, 95)) if len(v) else float("nan")


def current_training_values() -> dict:
    """지금 학습 값: train_phase_net.py의 TRAIN_ERRORS, env_cfg.py student 높이맵 GaussianNoiseCfg std."""
    out = {}
    try:
        with open(os.path.join(REPO, "scripts", "train_phase_net.py"), encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TRAIN_ERRORS" for t in node.targets):
                out.update(eval(compile(ast.Expression(node.value), "TRAIN_ERRORS", "eval"), {"math": math}))
    except (OSError, SyntaxError):
        pass
    try:
        with open(os.path.join(REPO, "src", "RLforHexapod", "tasks", "walk", "config", "hexapod", "env_cfg.py"),
                  encoding="utf-8") as f:
            m = re.search(r"height_map = ObsTerm\(.*?GaussianNoiseCfg\(std=([0-9.]+)\)", f.read())
        if m:
            out["height_noise_std"] = float(m.group(1))
    except OSError:
        pass
    return out


def summarize(results: list[dict], margin: float, climb_length: float) -> dict:
    statics = [r for r in results if r["kind"] == "static"]
    walks = [r for r in results if r["kind"] == "walk"]
    static_bias = _p95_abs(r["x_bias"] for r in statics)
    drift_per_m = _p95_abs(r["drift_per_m"] for r in walks)
    x_bias = static_bias + (drift_per_m * climb_length if np.isfinite(drift_per_m) else 0.0)
    measured = {
        "lidar_x_bias": x_bias,
        "lidar_x_noise": _p95_abs(r["x_noise"] for r in statics),
        "lidar_mount_tilt": _p95_abs(max(abs(r["pitch"]), abs(r["roll"])) for r in statics),
        "lidar_range_bias": _p95_abs(r["range_bias"] for r in statics),
        "height_noise_std": _p95_abs(r["height_noise"] for r in statics),
        "lidar_dropout": _p95_abs(r["dropout"] for r in statics),
        "lidar_period": float(np.nanmedian([r["period"] for r in statics])) if statics else float("nan"),
    }
    suggested = {k: (v if k == "lidar_period" else v * margin) for k, v in measured.items()}

    # map_mode: trial별 지도 밀림과 몸통 x 편향의 관계 (원점을 지나는 기울기)
    pairs = [(r["x_bias"], r["shift_mean"]) for r in statics
             if r["shift_observable"] and np.isfinite(r["x_bias"]) and np.isfinite(r["shift_mean"])]
    map_mode = {"trials": len(pairs)}
    if len(pairs) >= 3:
        xb, sh = np.array(pairs).T
        x_rms = float(np.sqrt((xb ** 2).mean()))
        slope = float((xb * sh).sum() / (xb * xb).sum()) if x_rms > 0 else float("nan")
        # 밀림은 2 cm 격자라 ±1 cm까지만 보인다. "밀림 = x 편향"과 "밀림 = 0" 중 어느 쪽에 더 가까운지로 정한다
        to_acc = float(np.sqrt(((sh - xb) ** 2).mean()))
        to_frame = float(np.sqrt((sh ** 2).mean()))
        if x_rms < 0.005:
            verdict = "몸통 x 편향이 5 mm보다 작아 2 cm 격자로는 구분할 수 없다 (어느 쪽이든 차이가 작다)"
        elif to_acc < 0.6 * to_frame:
            verdict = "accumulated (지금 설정과 같음)"
        elif to_frame < 0.6 * to_acc:
            verdict = "per_frame"
        else:
            verdict = "둘 다 아님: 지도 밀림과 몸통 x 오차를 따로 두도록 sensor_model을 고칠 것"
        map_mode.update(slope=slope, rms_to_accumulated=to_acc, rms_to_per_frame=to_frame, x_bias_rms=x_rms,
                        verdict=verdict)
    return {"static_trials": len(statics), "walk_trials": len(walks), "static_x_bias": static_bias,
            "drift_per_m": drift_per_m, "climb_length": climb_length, "margin": margin,
            "measured": measured, "suggested": suggested, "current": current_training_values(), "map_mode": map_mode}


# ----- 출력 -----
UNITS = {  # 항목: (표시 단위, 배율, 설명)
    "lidar_x_bias": ("mm", 1e3, "x 편향 (켤 때마다)"),
    "lidar_x_noise": ("mm", 1e3, "x 노이즈 std"),
    "lidar_mount_tilt": ("deg", 180 / math.pi, "높이맵 기울기"),
    "lidar_range_bias": ("mm", 1e3, "높이 편향"),
    "height_noise_std": ("mm", 1e3, "칸 노이즈 std"),
    "lidar_dropout": ("%", 100, "갱신 안 된 칸"),
    "lidar_period": ("ms", 1e3, "갱신 주기"),
}


def _fmt(v, scale, width=8, digits=1):
    return f"{v * scale:>{width}.{digits}f}" if v is not None and np.isfinite(v) else f"{'-':>{width}}"


def print_report(results: list[dict], summary: dict):
    statics = [r for r in results if r["kind"] == "static"]
    walks = [r for r in results if r["kind"] == "walk"]
    if statics:
        print("\n[static trial]  단위: mm, deg, %")
        print(f"{'이름':<16}{'true_x':>8}{'프레임':>7}{'x편향':>8}{'x노이즈':>8}{'rho1':>6}{'밀림':>8}{'높이편향':>8}"
              f"{'pitch':>7}{'roll':>7}{'칸노이즈':>8}{'모서리':>7}{'구멍':>7}{'안보임':>7}{'주기ms':>8}")
        for r in statics:
            print(f"{r['name']:<16}{_fmt(r['true_x'], 1e3)}{r['frames']:>7}{_fmt(r['x_bias'], 1e3)}"
                  f"{_fmt(r['x_noise'], 1e3)}{_fmt(r['x_rho1'], 1, 6, 2)}{_fmt(r['shift_mean'], 1e3)}"
                  f"{_fmt(r['range_bias'], 1e3)}{_fmt(r['pitch'], 180 / math.pi, 7, 2)}"
                  f"{_fmt(r['roll'], 180 / math.pi, 7, 2)}{_fmt(r['height_noise'], 1e3)}"
                  f"{_fmt(r['edge_miss'], 100, 7)}{_fmt(r['dropout'], 100, 7)}{_fmt(r['unseen'], 100, 7)}"
                  f"{_fmt(r['period'], 1e3)}")
    if walks:
        print("\n[walk trial]  단위: mm")
        print(f"{'이름':<16}{'시작':>8}{'끝':>8}{'시작오차':>9}{'끝오차':>8}{'드리프트':>9}{'mm/m':>8}{'놓침%':>7}")
        for r in walks:
            print(f"{r['name']:<16}{_fmt(r['true_x'], 1e3)}{_fmt(r['true_x_end'], 1e3)}{_fmt(r['x_err_start'], 1e3, 9)}"
                  f"{_fmt(r['x_err_end'], 1e3)}{_fmt(r['drift'], 1e3, 9)}{_fmt(r['drift_per_m'], 1e3)}"
                  f"{_fmt(r['lost'], 100, 7)}")

    m, s, cur = summary["measured"], summary["suggested"], summary["current"]
    print(f"\n[학습 설정 제안]  측정 = trial들의 95% (|값|), 제안 = 측정 × {summary['margin']}")
    if summary["walk_trials"]:
        print(f"  x 편향 = 정지 {summary['static_x_bias'] * 1e3:.1f} mm + 드리프트 "
              f"{summary['drift_per_m'] * 1e3:.1f} mm/m × 오르는 거리 {summary['climb_length']} m")
    print(f"{'항목':<20}{'설명':<20}{'측정':>8}{'제안':>8}{'지금':>8}  단위  판정")
    for k, (unit, scale, desc) in UNITS.items():
        now = cur.get(k)
        if k == "lidar_period":
            now = 0.0 if now is None else now  # 학습 기본값 0 = 매 step(0.02 s) 갱신
            verdict = "같음" if abs(now - s[k]) < 0.01 else "다름: 학습/평가에 측정값을 넣을 것"
        elif now is None or not np.isfinite(s[k]):
            verdict = ""
        else:
            verdict = "키울 것" if s[k] > now * 1.05 else "충분"
        print(f"{k:<20}{desc:<20}{_fmt(m[k], scale)}{_fmt(s[k], scale)}{_fmt(now, scale)}  {unit:<5} {verdict}")
    mm = summary["map_mode"]
    if "slope" in mm:
        print(f"\n[map_mode] trial {mm['trials']}개: 지도 밀림 ≈ {mm['slope']:.2f} × 몸통 x 편향, "
              f"차이 rms accumulated {mm['rms_to_accumulated'] * 1e3:.1f} mm / per_frame "
              f"{mm['rms_to_per_frame'] * 1e3:.1f} mm → {mm['verdict']}")
    else:
        print(f"\n[map_mode] 모서리가 높이맵 안에 있는 static trial이 {mm['trials']}개라 판정 못 함 (3개 이상 필요)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="trial .npz 파일 또는 폴더")
    p.add_argument("--warmup", type=float, default=3.0, help="static: 앞에서 버릴 시간 [s] (지도가 쌓이는 동안)")
    p.add_argument("--still", type=float, default=2.0, help="walk: 시작·끝 정지 구간 [s]")
    p.add_argument("--margin", type=float, default=1.5, help="제안 = 측정 × margin")
    p.add_argument("--climb_length", type=float, default=1.2, help="계단 하나를 오르며 걷는 거리 [m]")
    p.add_argument("--json", default=None, help="결과를 이 json 파일로 저장")
    args = p.parse_args()

    files = []
    for path in args.paths:
        files += sorted(glob.glob(os.path.join(path, "*.npz"))) if os.path.isdir(path) else [path]
    if not files:
        sys.exit("trial 파일이 없다")
    results = []
    for path in files:
        trial = load_trial(path)
        kind = trial["meta"]["kind"]
        results.append(analyze_static(trial, args.warmup) if kind == "static" else analyze_walk(trial, args.still))
    summary = summarize(results, args.margin, args.climb_length)
    print_report(results, summary)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"trials": results, "summary": summary}, f, ensure_ascii=False, indent=2)
        print(f"\n저장: {args.json}")


if __name__ == "__main__":
    main()
