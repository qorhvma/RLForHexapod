"""A-Phase 계단 student 체크포인트를 걷기·계단 통합 A-Phase task(Hexapod-Student-A-Phase)의 시작점으로 바꾼다.

통합 student 관측은 A-Phase 관측의 단계 칸(7) 앞에 조종기 명령(4)이 끼어 있다. 그 자리에 입력 4칸을 넣는다.
    - GRU 입력 가중치: 새 칸은 0. 그래서 처음에는 명령과 상관없이 A-Phase와 똑같이 움직인다.
    - 관측 정규화: 새 칸은 명령 분포로 계산한 평균·표준편차 (아래 COMMAND_*). 0/1로 두면 높이 명령이
      0.2m 근처에 몰려 있어 정규화 뒤 값이 0.2/1로 작게 들어가 배우기 느리다.
    - optimizer: 같은 설정으로 새로 시작한다 (모양이 바뀐 가중치의 이전 기울기 기록은 쓸 수 없다).
teacher는 --teachers(merge_teachers.py로 합친 파일)의 걷기·계단 teacher를 그대로 담는다.
계단 teacher는 A-Phase student를 만든 teacher와 같아야 한다 (다르면 멈춘다).

    uv run python scripts/merge_teachers.py --walk logs/teachers.pt --stair <A-Phase를 만든 계단 model.pt> --out logs/teachers_mixed_phase.pt
    uv run python scripts/expand_student.py --student <A_phase model_*.pt> --teachers logs/teachers_mixed_phase.pt --out logs/mixed_phase_init.pt
    uv run isaaclab train --rl_library rsl_rl --task Hexapod-Student-A-Phase --num_envs 4096 --checkpoint logs/mixed_phase_init.pt
"""

import argparse
from pathlib import Path

import torch

A_PHASE_OBS_DIM = 278  # Student-A 관측 271 + 단계 7
PHASE_OBS_SIZE = 7
COMMAND_DIM = 4  # 속도 x, y, yaw + 몸통 높이 (mdp.student_command)

# 통합 env(stair_ratio 0.5)의 명령 분포. 계단 env는 고정값 (0.2, 0, 0, 0.24),
# 걷기 env는 env_cfg.CommandCfg 범위의 균등 분포 (10%는 속도 0, 높이는 늘 0.17~0.24)
#   vx: 계단 0.2 / 걷기 평균 0, 분산 0.9 × 0.4²/12 = 0.012 → 평균 0.1, 분산 0.016
#   vy: 걷기만 분산 0.012 → 분산 0.006
#   yaw: 걷기만 분산 0.9 × 1.0²/12 = 0.075 → 분산 0.0375
#   높이: 계단 0.24 / 걷기 평균 0.205, 분산 0.07²/12 → 평균 0.2225, 분산 0.00051
COMMAND_MEAN = [0.1, 0.0, 0.0, 0.2225]
COMMAND_VAR = [0.016, 0.006, 0.0375, 0.00051]


def insert_columns(t: torch.Tensor, at: int, values: torch.Tensor) -> torch.Tensor:
    """마지막 축의 at 자리에 values (COMMAND_DIM,)를 넣는다."""
    fill = values.to(t.dtype).expand(*t.shape[:-1], COMMAND_DIM)
    return torch.cat([t[..., :at], fill, t[..., at:]], dim=-1)


def expand_student(student: dict) -> dict:
    """A-Phase student state_dict → 통합 A-Phase student state_dict."""
    weight_key = "rnn.rnn.weight_ih_l0"
    if student[weight_key].shape[-1] != A_PHASE_OBS_DIM:
        raise ValueError(f"A-Phase student의 입력은 {A_PHASE_OBS_DIM}차원이어야 하는데 {student[weight_key].shape[-1]}차원입니다.")
    at = A_PHASE_OBS_DIM - PHASE_OBS_SIZE
    mean, var = torch.tensor(COMMAND_MEAN), torch.tensor(COMMAND_VAR)
    fills = {
        weight_key: torch.zeros(COMMAND_DIM),
        "obs_normalizer._mean": mean,
        "obs_normalizer._var": var,
        "obs_normalizer._std": var.sqrt(),
    }
    out = {}
    for key, value in student.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.shape[-1] == A_PHASE_OBS_DIM:
            if key not in fills:
                raise ValueError(f"입력 크기를 가진 예상 밖의 항목: {key} {tuple(value.shape)}")
            value = insert_columns(value, at, fills[key])
        out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description="A-Phase 계단 student → 통합 A-Phase student 시작점")
    parser.add_argument("--student", required=True, help="A-Phase 증류 체크포인트 model_*.pt")
    parser.add_argument("--teachers", required=True, help="merge_teachers.py로 합친 걷기·계단 teacher 파일")
    parser.add_argument("--out", required=True, help="저장할 파일")
    args = parser.parse_args()

    checkpoint = torch.load(args.student, weights_only=False, map_location="cpu")
    if "student_state_dict" not in checkpoint:
        raise ValueError(f"{args.student}: 증류 체크포인트가 아닙니다 (student_state_dict 없음).")
    teachers = torch.load(args.teachers, weights_only=False, map_location="cpu")["actor_state_dict"]
    if not any(key.startswith("walk.") for key in teachers):
        raise ValueError(f"{args.teachers}: 합친 teacher 파일이 아닙니다 (walk.* 없음). merge_teachers.py로 만드세요.")

    # 계단 teacher가 바뀌면 student가 배운 동작과 정답이 어긋난다
    stair = {key[len("stair."):]: value for key, value in teachers.items() if key.startswith("stair.")}
    old_teacher = checkpoint["teacher_state_dict"]
    if stair.keys() != old_teacher.keys() or any(
        not torch.equal(stair[key], old_teacher[key]) for key in stair if key != "obs_normalizer.count"
    ):
        raise ValueError(
            f"{args.teachers}의 계단 teacher가 {args.student}를 만든 teacher와 다릅니다. "
            "같은 계단 model.pt로 merge_teachers.py를 다시 실행하세요."
        )

    optimizer = checkpoint["optimizer_state_dict"]
    out = {
        "student_state_dict": expand_student(checkpoint["student_state_dict"]),
        "teacher_state_dict": teachers,
        # 설정(학습률 등)만 두고 기록은 비운다
        "optimizer_state_dict": {"state": {}, "param_groups": optimizer["param_groups"]},
        "iter": 0,
        "infos": None,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"student: {args.student} (iter {checkpoint.get('iter')}) → 입력 {A_PHASE_OBS_DIM + COMMAND_DIM}차원")
    print(f"teacher: {args.teachers}")
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
