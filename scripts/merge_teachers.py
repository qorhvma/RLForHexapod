"""걷기·계단 teacher 체크포인트를 통합 student 증류용 파일 하나로 합친다.

    uv run python scripts/merge_teachers.py --walk <걷기 model.pt> --stair <계단 model.pt> --out <teachers.pt>

--walk/--stair에는 이미 합친 파일(teachers.pt)도 줄 수 있다. 그 안의 해당 teacher만 꺼내 쓴다.
걷기 teacher 원본이 없는 PC에서 계단 teacher만 바꿀 때 쓴다:

    uv run python scripts/merge_teachers.py --walk logs/teachers.pt --stair <계단 model.pt> --out <새 파일>
"""

import argparse
from pathlib import Path

import torch

WALK_OBS_DIM = 67
STAIR_OBS_DIM = 92


def load_actor(path: str, expected_obs_dim: int, name: str) -> dict:
    """PPO 체크포인트(또는 합친 파일의 name 쪽)에서 actor를 꺼내고, 입력 크기로 맞는 teacher인지 확인한다."""
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    if "actor_state_dict" not in checkpoint:
        raise ValueError(f"{path}: PPO 체크포인트가 아닙니다 (actor_state_dict 없음).")
    actor = checkpoint["actor_state_dict"]
    prefix = f"{name}."
    if any(key.startswith(prefix) for key in actor):  # 합친 파일
        actor = {key[len(prefix):]: value for key, value in actor.items() if key.startswith(prefix)}
    obs_dim = actor["mlp.0.weight"].shape[1]
    if obs_dim != expected_obs_dim:
        raise ValueError(f"{path}: {name} teacher의 입력은 {expected_obs_dim}차원이어야 하는데 {obs_dim}차원입니다.")
    print(f"{name}: {path} (iter {checkpoint.get('iter')}, 입력 {obs_dim}차원)")
    return actor


def main():
    parser = argparse.ArgumentParser(description="걷기·계단 teacher 체크포인트 합치기")
    parser.add_argument("--walk", required=True, help="걷기 teacher model_*.pt")
    parser.add_argument("--stair", required=True, help="계단 teacher model_*.pt")
    parser.add_argument("--out", required=True, help="저장할 파일")
    args = parser.parse_args()

    merged = {}
    for prefix, path, obs_dim in (("walk", args.walk, WALK_OBS_DIM), ("stair", args.stair, STAIR_OBS_DIM)):
        for key, value in load_actor(path, obs_dim, prefix).items():
            merged[f"{prefix}.{key}"] = value

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # rsl_rl은 actor_state_dict가 있으면 teacher만 불러오고 student는 새로 시작한다
    torch.save({"actor_state_dict": merged, "iter": 0, "infos": None}, args.out)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
