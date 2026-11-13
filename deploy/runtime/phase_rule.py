"""규칙 단계 추정기 (numpy, 로봇 한 대). 시뮬레이션 mdp/phase_estimator.py + StairTracker.check_phase와 같은 판정이다.

단계 추정 신경망의 입력 phase_rule(40) = stair_rule_features, phase_timer(1) = stair_rule_phase_time을 만든다.
표(PHASES)와 판정 기준(StairGaitCfg)은 robot_io.json의 phase_rule에서 읽는다 (내보낼 때 학습 코드에서 복사됨).
판정 설정은 단계 신경망 학습 때의 기본값만 지원한다: 지연 0, by_contact 끔, catch_up 끔, slack 0.

    rule = PhaseRule(io)                 # io = robot_io.json
    rule.reset()                         # 계단 앞에서 동작을 시작할 때 (단계 0)
    m = Measurement(...)                 # 매 step, 계단 좌표 (아래)
    rule.update(m)                       # 단계 진행
    features, time = rule.features(m), rule.time_feature()

계단 좌표 = 시뮬레이션 env 원점 기준: 첫 수직면이 x = x0 (0.5 m), 계단 가운데 선이 y = 0, 바닥이 z = 0.
"""

from dataclasses import dataclass

import numpy as np

RULE_X_CLIP = (-0.3, 0.3)   # mdp/stair_reward.py와 같은 값
RULE_Z_CLIP = (-0.1, 0.2)
RULE_TIME_MAX = 10.0
RULE_TIME_SCALE = 2.0


@dataclass
class Measurement:
    """규칙 추정기가 한 step에 보는 값 (시뮬레이션 PhaseMeasurement와 같은 뜻)."""

    body_x: float            # 몸통 x, 계단 좌표 [m]
    tips: np.ndarray         # (6, 3) 발끝 추정점, 계단 좌표 [m], FOOT_ORDER 순서
    in_contact: np.ndarray   # (6,) bool 발 접촉
    front_coxa: np.ndarray   # (2,) 앞다리 coxa 각도 [rad], front_coxa_joints 순서 (L4, L3)
    speed: float             # |(선속도, 각속도)| 6차원 노름 (시뮬레이션 root_vel_w의 노름) [m/s, rad/s 섞임]


class PhaseRule:
    def __init__(self, io: dict, dt: float | None = None):
        cfg = io["phase_rule"]
        est = cfg["estimator"]
        unsupported = {k: v for k, v in est.items() if k in ("delay", "by_contact", "catch_up", "advance_slack",
                                                             "over_slack", "fold_slack", "contact_confirm") and v}
        if unsupported:
            raise ValueError(f"이 런타임은 기본 판정 설정만 지원합니다: {unsupported}")
        st = io["stairs_trained"]
        self.x0, self.tread, self.step_height = st["x0"], st["tread"], st["step_height"]
        self.num_steps, self.top_length, self.width = st["num_steps"], st["top_length"], st["width"]
        g = cfg["gait"]
        self.g = g
        self.dt = float(io["policy_dt"] if dt is None else dt)
        phases = cfg["phases"]
        self.num_phases = len(phases)
        self.ADVANCE, self.MOVE = cfg["kinds"]["advance"], cfg["kinds"]["move"]
        self.kind = np.array([p["kind"] for p in phases])
        self.table_body_x = np.array([p["body_x"] for p in phases])
        self.table_fold = np.array([p["fold"] for p in phases])
        self.table_moving = np.zeros((self.num_phases, 6), bool)
        self.table_placed = np.zeros((self.num_phases, 6))
        placed = []
        for k, p in enumerate(phases):
            self.table_moving[k, p["feet"]] = True
            self.table_placed[k, placed] = 1.0
            if p["kind"] == self.MOVE:
                placed = placed + list(p["feet"])
        self.final_advance = self.num_phases * self.num_steps
        self.settle_move = self.final_advance + 1
        self.done_move = self.settle_move + 1
        self.front_parallel = np.array([g["front_parallel_coxa"], -g["front_parallel_coxa"]])
        self.reset()

    # ----- 상태 -----
    def reset(self, move: int = 0):
        self.move = int(move)
        self.best_body_x = 0.0
        self.need_init = True
        self.steady_time = 0.0
        self.phase_time = 0.0

    # ----- 계단 형상 (mdp.stairs_terrain_height) -----
    def terrain(self, x, y):
        xr = np.asarray(x, float) - self.x0
        length = (self.num_steps - 1) * self.tread + self.top_length
        on = (xr >= 0.0) & (xr < length) & (np.abs(y) <= self.width / 2)
        k = np.minimum(np.floor(xr / self.tread) + 1, self.num_steps)
        return np.where(on, k * self.step_height, 0.0)

    def phase_and_level(self, move=None):
        climbing = min(self.move if move is None else move, self.final_advance)
        return climbing % self.num_phases, climbing // self.num_phases

    def phase_one_hot(self, move=None):
        move = self.move if move is None else move
        phase, _ = self.phase_and_level(move)
        out = np.zeros(self.num_phases + 1)
        out[self.num_phases if move >= self.settle_move else phase] = 1.0
        return out

    def riser_and_top(self, level):
        return self.x0 + level * self.tread, (level + 1) * self.step_height

    # ----- 판정 (StairTracker.check_phase) -----
    def check(self, m: Measurement) -> tuple[bool, float, float]:
        """(이번 step에 단계 완료, 새 신기록, 새 멈춘 시간). 상태를 바꾸지 않는다."""
        g, move = self.g, self.move
        phase, level = self.phase_and_level()
        riser, top = self.riser_and_top(level)
        in_stairs = move < self.final_advance
        moving = self.table_moving[phase] & in_stairs
        kind = self.kind[phase]
        advancing = kind == self.ADVANCE and move <= self.final_advance
        moving_pair = kind == self.MOVE and in_stairs
        settling = move == self.settle_move

        tips = np.asarray(m.tips, float)
        terrain = self.terrain(tips[:, 0], tips[:, 1])
        near_ground = np.abs(tips[:, 2] - terrain - g["tip_height"]) < g["contact_tol"]

        limit_x = riser + self.table_body_x[phase]
        capped = min(m.body_x, limit_x)
        start_best = capped if self.need_init else self.best_body_x
        feet_up = bool(np.all((tips[:, 2] >= top) | ~moving))
        body_counts = advancing or (moving_pair and feet_up)
        new_best = max(start_best, capped) if body_counts else start_best

        fold = np.minimum(np.asarray(m.front_coxa, float) / self.front_parallel, 1.0)
        needs_fold = in_stairs and bool(self.table_fold[phase])

        contact = np.asarray(m.in_contact, bool)
        advance_done = advancing and new_best >= limit_x - g["advance_tol"]
        over_riser = tips[:, 0] >= riser + g["over_margin"]
        on_target = over_riser & (terrain >= top - 1e-3) & near_ground & contact
        move_done = moving_pair and bool(np.all(on_target | ~moving))

        placed_height = (level + self.table_placed[phase]) * self.step_height
        feet_kept = ((terrain >= placed_height - 1e-3) & near_ground) | moving
        all_kept = bool(feet_kept.all())

        steady = settling and all_kept and bool(contact.all()) and m.speed < g["speed_tol"]
        new_steady = self.steady_time + self.dt if steady else 0.0
        settle_done = settling and new_steady >= g["hold_time"]

        fold_required = g["front_fold_required"] if kind == self.ADVANCE else g["front_fold_hold"]
        fold_ok = bool(np.all(fold >= fold_required)) or not needs_fold
        done = ((advance_done or move_done) and all_kept and fold_ok) or settle_done
        return done, new_best, new_steady

    def update(self, m: Measurement):
        """단계를 진행시킨다 (PhaseEstimator.update, 기본 설정)."""
        done, self.best_body_x, self.steady_time = self.check(m)
        self.move += int(done)
        self.need_init = done
        self.phase_time = 0.0 if done else self.phase_time + self.dt

    # ----- 단계 신경망 입력 -----
    def features(self, m: Measurement) -> np.ndarray:
        """stair_rule_features (40,)."""
        g = self.g
        phase, level = self.phase_and_level()
        riser, _ = self.riser_and_top(level)
        tips = np.asarray(m.tips, float)
        terrain = self.terrain(tips[:, 0], tips[:, 1])
        foot_level = terrain / self.step_height - level
        past_riser = np.clip(tips[:, 0] - (riser + g["over_margin"]), *RULE_X_CLIP)
        height = np.clip(tips[:, 2] - terrain - g["tip_height"], *RULE_Z_CLIP)
        fold = np.minimum(np.asarray(m.front_coxa, float) / self.front_parallel, 1.0)
        limit_x = riser + self.table_body_x[phase]
        level_hot = np.zeros(self.num_steps + 1)
        level_hot[level] = 1.0
        return np.concatenate([
            self.phase_one_hot(), level_hot,
            np.clip(np.round(foot_level), -2.0, 2.0), past_riser, height, np.asarray(m.in_contact, float),
            fold, [np.clip(m.body_x - limit_x, *RULE_X_CLIP)], [np.clip(self.best_body_x - limit_x, *RULE_X_CLIP)],
            [min(self.steady_time / g["hold_time"], 1.0)],
        ]).astype(np.float32)

    def time_feature(self) -> float:
        """stair_rule_phase_time (1,)."""
        return min(self.phase_time, RULE_TIME_MAX) / RULE_TIME_SCALE
