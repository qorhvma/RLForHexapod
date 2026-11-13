"""실물 라이다 높이맵: 시뮬레이션의 mdp.stairs_height_map, mdp.lidar_body_x와 같은 정의를 numpy로 만든다.

torch·Isaac Lab 없이 돈다 (라즈베리파이, 분석 PC).
record_ros2.py(기록), analyze.py(분석), synth.py(합성 시험)가 같이 쓴다.

좌표
    지도(map)    Point-LIO 월드 좌표. z가 위 (Point-LIO의 gravity_align을 켠다)
    계단(stairs) 첫 수직면이 x=0, 계단 쪽이 +x, 계단 가운데 선이 y=0
    s축          기록 시작 때 몸통 위치와 앞 방향으로 정한 지도 위 직선. 수직면은 이 축 위에서 찾는다
                 (로봇이 계단을 바라보고 시작하므로 s축 ≈ 계단 x축)
    격자(grid)   몸통 기준. 몸통 heading만 돌리고 기울기는 무시한다 (시뮬레이션과 같음)
높이맵 값 = 몸통 z - 지형 z (시뮬레이션과 같은 부호, 단위 m)
"""

from dataclasses import asdict, dataclass, field

import numpy as np

# 시뮬레이션 student 높이맵과 같은 격자 (env_cfg.py: x_range (-0.24, 0.56), y_range (-0.2, 0.2), spacing (0.02, 0.1))
GRID_X = -0.24 + 0.02 * np.arange(41)
GRID_Y = -0.2 + 0.1 * np.arange(5)
GX, GY = np.meshgrid(GRID_X, GRID_Y, indexing="ij")  # (41, 5). flatten하면 시뮬레이션 관측 순서와 같다


# ----- 계단 형상 -----
@dataclass
class Stairs:
    """실측한 계단 형상 (계단 좌표). 단마다 다를 수 있어 목록으로 둔다."""

    riser_x: list[float] = field(default_factory=lambda: [0.0, 0.3, 0.6])  # 각 수직면 x [m]. 첫 값이 0
    top_z: list[float] = field(default_factory=lambda: [0.16, 0.32, 0.48])  # 각 단 윗면 높이 (바닥 기준) [m]
    end_x: float = 1.4  # 꼭대기 끝 x [m]
    width: float = 1.0  # 계단 폭 [m]

    @classmethod
    def uniform(cls, step_height=0.16, tread=0.3, num_steps=3, top_length=0.8, width=1.0) -> "Stairs":
        """시뮬레이션 StairsGeometryCfg와 같은 고른 계단."""
        riser_x = [tread * i for i in range(num_steps)]
        return cls(riser_x, [step_height * (i + 1) for i in range(num_steps)],
                   riser_x[-1] + top_length, width)

    @classmethod
    def from_dict(cls, d: dict) -> "Stairs":
        return cls(list(d["riser_x"]), list(d["top_z"]), float(d["end_x"]), float(d["width"]))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def edges(self) -> np.ndarray:
        """높이가 바뀌는 x (수직면들 + 꼭대기 끝)."""
        return np.array([*self.riser_x, self.end_x])

    def height(self, x, y) -> np.ndarray:
        """계단 좌표 (x, y)의 지형 높이. 계단 밖은 바닥(0). stairs_terrain_height와 같은 정의."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        idx = np.searchsorted(np.asarray(self.riser_x), x, side="right") - 1
        top = np.asarray(self.top_z)[np.clip(idx, 0, None)]
        on = (idx >= 0) & (x < self.end_x) & (np.abs(y) <= self.width / 2)
        return np.where(on, top, 0.0)


def truth_height_map(stairs: Stairs, x, y, z, yaw=0.0) -> np.ndarray:
    """몸통이 계단 좌표 (x, y, z), heading yaw [rad]에 있을 때의 참 높이맵 (41, 5)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return z - stairs.height(x + c * GX - s * GY, y + s * GX + c * GY)


# ----- 자세 -----
def quat_to_matrix(q) -> np.ndarray:
    """(x, y, z, w) 쿼터니언 (ROS 순서) → 회전 행렬."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rpy_to_matrix(roll, pitch, yaw) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def matrix_to_rpy(r) -> tuple[float, float, float]:
    """회전 행렬 → (roll, pitch, yaw). yaw는 앞 방향(x축)의 heading = 시뮬레이션과 같은 정의."""
    return (float(np.arctan2(r[2, 1], r[2, 2])), float(-np.arcsin(np.clip(r[2, 0], -1.0, 1.0))),
            float(np.arctan2(r[1, 0], r[0, 0])))


# ----- 지도 -----
class TerrainMap:
    """지도 좌표의 2D 칸(기본 1 cm)마다 지형 윗면 높이를 쌓는다.

    프레임마다 칸의 가장 높은 점에서 band 안의 점만 평균한다 (아래 바닥 점이 섞이지 않게).
    한 프레임에서 칸 안 점의 높이 폭이 face_spread보다 크면 수직면 점이 든 칸으로 보고 이번 프레임은 쓰지 않는다
    (수직면 점이 모서리 앞 칸에 들어가 윗단처럼 보이면 수직면이 앞당겨 보인다).
    기존 값보다 band 넘게 높으면 새 값으로 바꾸고 (가려졌던 윗면이 보임), 아니면 점 수로 가중 평균한다.
    가중치는 memory에서 멈춰 오래된 값이 천천히 잊힌다."""

    def __init__(self, center_xy, size=6.0, res=0.01, band=0.03, min_points=2, memory=30.0, face_spread=0.06):
        self.res, self.band, self.min_points, self.memory = res, band, min_points, memory
        self.face_spread = face_spread
        self.n = int(round(size / res))
        self.origin = np.asarray(center_xy, float) - size / 2
        self.z = np.full(self.n * self.n, np.nan)
        self.weight = np.zeros(self.n * self.n)
        self.fresh = np.zeros(self.n * self.n, bool)  # 이번 프레임에 점이 들어온 칸

    def _index(self, xy):
        ij = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(np.int64)
        inside = np.all((ij >= 0) & (ij < self.n), axis=-1)
        return ij, inside

    def add_frame(self, points: np.ndarray):
        """points: 지도 좌표 점 (N, 3)."""
        self.fresh[:] = False
        points = points[np.all(np.isfinite(points), axis=1)]
        ij, inside = self._index(points[:, :2])
        if not inside.any():
            return
        flat = ij[inside, 0] * self.n + ij[inside, 1]
        z = points[inside, 2]
        cells, inv = np.unique(flat, return_inverse=True)
        zmax = np.full(len(cells), -np.inf)
        np.maximum.at(zmax, inv, z)
        zmin = np.full(len(cells), np.inf)
        np.minimum.at(zmin, inv, z)
        top = z >= zmax[inv] - self.band
        total = np.bincount(inv[top], z[top], minlength=len(cells))
        count = np.bincount(inv[top], minlength=len(cells)).astype(float)
        ok = (count >= self.min_points) & (zmax - zmin <= self.face_spread)
        cells, value, count = cells[ok], total[ok] / count[ok], count[ok]
        old, w = self.z[cells], self.weight[cells]
        replace = np.isnan(old) | (value > old + self.band)
        self.z[cells] = np.where(replace, value, (np.nan_to_num(old) * w + value * count) / (w + count))
        self.weight[cells] = np.minimum(np.where(replace, count, w + count), self.memory)
        self.fresh[cells] = True

    def sample(self, xy, radius=1) -> tuple[np.ndarray, np.ndarray]:
        """지점마다 주변 (2·radius+1)² 칸의 평균 높이와 이번 프레임 갱신 여부. 값이 없으면 nan."""
        xy = np.asarray(xy, float)
        shape = xy.shape[:-1]
        ij, _ = self._index(xy.reshape(-1, 2))
        d = np.arange(-radius, radius + 1)
        di, dj = np.meshgrid(d, d, indexing="ij")
        ni, nj = ij[:, 0:1] + di.ravel(), ij[:, 1:2] + dj.ravel()  # (M, K)
        inside = (ni >= 0) & (ni < self.n) & (nj >= 0) & (nj < self.n)
        flat = np.where(inside, ni * self.n + nj, 0)
        z = np.where(inside, self.z[flat], np.nan)
        valid = ~np.isnan(z)
        count = valid.sum(axis=1)
        with np.errstate(invalid="ignore"):
            mean = np.where(count > 0, np.where(valid, z, 0.0).sum(axis=1) / np.maximum(count, 1), np.nan)
        fresh = (self.fresh[flat] & inside).any(axis=1)
        return mean.reshape(shape), fresh.reshape(shape)


# ----- 수직면 찾기 -----
class StairLocator:
    """s축 위 높이 단면에서 수직면을 찾는다. 단마다 (아래 단 + 윗 단)/2 높이를 처음 넘는 곳 (칸 사이는 선형 보간)."""

    def __init__(self, origin_xy, yaw, stairs: Stairs, s_range=(-1.0, 3.0), ds=0.01, lateral=0.2):
        self.origin = np.asarray(origin_xy, float)
        self.dir = np.array([np.cos(yaw), np.sin(yaw)])
        self.perp = np.array([-self.dir[1], self.dir[0]])
        self.stairs = stairs
        self.s = np.arange(s_range[0], s_range[1] + 1e-9, ds)
        offsets = np.linspace(-lateral, lateral, 5)
        self.points = self.origin + self.s[:, None, None] * self.dir + offsets[None, :, None] * self.perp

    def s_of(self, xy) -> float:
        return float((np.asarray(xy, float) - self.origin) @ self.dir)

    def profile(self, tmap: TerrainMap) -> np.ndarray:
        z, _ = tmap.sample(self.points)  # (s 수, 옆 5줄)
        out = np.full(len(self.s), np.nan)
        rows = (~np.isnan(z)).sum(axis=1) >= 3
        if rows.any():
            out[rows] = np.nanmedian(z[rows], axis=1)
        return out

    def locate(self, tmap: TerrainMap) -> tuple[np.ndarray, float]:
        """(수직면 s 위치들, 바닥 z). 못 찾은 수직면은 nan."""
        p = self.profile(tmap)
        risers = np.full(len(self.stairs.riser_x), np.nan)
        valid = ~np.isnan(p)
        if valid.sum() < 10:
            return risers, np.nan
        floor = float(np.percentile(p[valid], 5))  # 가장 낮은 면 = 바닥 (계단 앞 바닥이 지도에 있다)
        s, p = self.s[valid], p[valid]
        start, prev = -np.inf, 0.0
        for k, top in enumerate(self.stairs.top_z):
            thr = floor + (prev + top) / 2
            above = p >= thr
            hits = np.flatnonzero(above[1:] & ~above[:-1] & (s[1:] > start)) + 1
            if len(hits) == 0:
                break
            i = hits[0]
            s0, s1, p0, p1 = s[i - 1], s[i], p[i - 1], p[i]
            risers[k] = s0 + (thr - p0) / (p1 - p0) * (s1 - s0) if p1 > p0 else s1
            start, prev = risers[k] + 0.05, top
        return risers, floor


@dataclass
class FrameResult:
    base: np.ndarray       # 몸통 지도 자세 (x, y, z, roll, pitch, yaw)
    s_base: float          # 몸통의 s축 위치
    risers: np.ndarray     # 수직면 s 위치들
    floor_z: float         # 지도에서 찾은 바닥 z
    body_x: float          # 첫 수직면 기준 몸통 x (= lidar_body_x 첫 값)
    height_map: np.ndarray  # (41, 5), 값이 없는 칸은 nan
    fresh: np.ndarray       # (41, 5) bool, 이번 프레임에 갱신된 칸


class HeightMapPipeline:
    """프레임(지도 좌표 점 + 몸통 자세)마다 지도를 쌓고 높이맵과 첫 수직면 기준 몸통 x를 낸다.
    s축은 첫 프레임의 몸통 위치·앞 방향으로 정한다 (로봇이 계단을 바라보고 시작해야 한다)."""

    def __init__(self, stairs: Stairs, map_size=6.0, map_res=0.01):
        self.stairs, self.map_size, self.map_res = stairs, map_size, map_res
        self.map = None
        self.locator = None

    def step(self, points: np.ndarray, base_pos, base_rot: np.ndarray) -> FrameResult:
        roll, pitch, yaw = matrix_to_rpy(base_rot)
        base_pos = np.asarray(base_pos, float)
        if self.map is None:
            self.map = TerrainMap(base_pos[:2] + 1.0 * np.array([np.cos(yaw), np.sin(yaw)]),
                                  self.map_size, self.map_res)
            self.locator = StairLocator(base_pos[:2], yaw, self.stairs)
        self.map.add_frame(points)
        c, s = np.cos(yaw), np.sin(yaw)
        xy = np.stack([base_pos[0] + c * GX - s * GY, base_pos[1] + s * GX + c * GY], axis=-1)
        z, fresh = self.map.sample(xy)
        risers, floor = self.locator.locate(self.map)
        s_base = self.locator.s_of(base_pos[:2])
        return FrameResult(np.array([*base_pos, roll, pitch, yaw]), s_base, risers, floor,
                           s_base - risers[0], base_pos[2] - z, fresh)


# ----- 기록 파일 -----
class TrialLog:
    """한 번 측정(trial)의 프레임들. npz 하나로 저장한다.

    meta (json): name, kind ("static" | "walk"), true_x, true_y, true_z, true_yaw_deg, true_x_end(walk만), stairs, note
        true_x: 첫 수직면 기준 몸통 원점 x [m] (계단 앞이면 음수), true_z: 바닥 기준 몸통 원점 높이 [m]
        walk는 시작에 true_x, 끝에 true_x_end에 정지해 있어야 한다"""

    def __init__(self, meta: dict):
        self.meta = meta
        self.rows = {k: [] for k in ("t_stamp", "t_recv", "base", "s_base", "risers", "floor_z", "body_x",
                                     "height_map", "fresh", "n_points")}

    def add(self, r: FrameResult, t_stamp: float, t_recv: float, n_points: int):
        for k, v in (("t_stamp", t_stamp), ("t_recv", t_recv), ("base", r.base), ("s_base", r.s_base),
                     ("risers", r.risers), ("floor_z", r.floor_z), ("body_x", r.body_x),
                     ("height_map", r.height_map), ("fresh", r.fresh), ("n_points", n_points)):
            self.rows[k].append(v)

    def __len__(self):
        return len(self.rows["t_stamp"])

    def save(self, path):
        import json
        np.savez_compressed(path, meta=json.dumps(self.meta, ensure_ascii=False),
                            **{k: np.asarray(v) for k, v in self.rows.items()})


def load_trial(path) -> dict:
    import json
    with np.load(path) as f:
        data = {k: f[k] for k in f.files}
    data["meta"] = json.loads(str(data["meta"]))
    return data
