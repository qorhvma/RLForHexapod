# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "onnxruntime"]
# ///
"""export_robot_policy.py로 만든 ONNX 파일이 시뮬레이션 기록(reference.npz)과 같은 값을 내는지 검사한다.

Isaac Lab·GPU 없이 CPU로 돈다. 로봇(Raspberry Pi)에서도 같은 방법으로 돌려 볼 수 있다.

    uv run scripts/verify_robot_policy.py deploy/mixed_phase_v1

검사 내용 (env마다 따로, 에피소드가 끝나면 기억을 0으로):
1. 단계 추정 신경망: 기록된 입력 net_x → logits가 기록과 같은지
2. 그 argmax one-hot이 기록된 student 관측의 단계 칸과 같은지
3. student: 기록된 관측 → actions가 기록과 같은지
로봇 쪽 구현도 같은 순서(신경망 → 단계 칸 채우기 → student)로 돌리면 된다.
"""

import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

TOL = 1e-4


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "deploy/mixed_phase_v1")
    io = json.loads((folder / "robot_io.json").read_text(encoding="utf-8"))
    ref = np.load(folder / "reference.npz")
    student = ort.InferenceSession(str(folder / io["student"]["file"]), providers=["CPUExecutionProvider"])
    net = ort.InferenceSession(str(folder / io["phase_net"]["file"]), providers=["CPUExecutionProvider"])
    h_student_shape = io["student"]["inputs"]["h_in"]
    h_net_shape = io["phase_net"]["inputs"]["h_in"]
    p0, p1 = io["phase_net"]["phase_slice_in_student_obs"]

    T, N = ref["actions"].shape[:2]
    worst_logits = worst_actions = 0.0
    phase_mismatch = 0
    for e in range(N):
        hs = np.zeros(h_student_shape, np.float32)
        hn = np.zeros(h_net_shape, np.float32)
        for t in range(T):
            logits, hn = net.run(None, {"x": ref["net_x"][t, e][None].astype(np.float32), "h_in": hn})
            worst_logits = max(worst_logits, float(np.abs(logits[0] - ref["logits"][t, e]).max()))
            obs = ref["student_obs"][t, e][None].astype(np.float32)
            phase = np.zeros(p1 - p0, np.float32)
            phase[int(np.argmax(logits[0]))] = 1.0
            phase_mismatch += int(not np.array_equal(phase, obs[0, p0:p1]))
            actions, hs = student.run(None, {"obs": obs, "h_in": hs})
            worst_actions = max(worst_actions, float(np.abs(actions[0] - ref["actions"][t, e]).max()))
            if ref["done"][t, e]:
                hs = np.zeros_like(hs)
                hn = np.zeros_like(hn)

    is_stair = ref["is_stair"]
    pred = ref["logits"].argmax(-1)
    agree = (pred == ref["true_phase"])[:, is_stair].mean() if is_stair.any() else float("nan")
    print(f"기록: {T} step × {N} env (계단 {int(is_stair.sum())}, 걷기 {int((~is_stair).sum())}), 끝난 에피소드 {int(ref['done'].sum())}")
    print(f"단계 신경망 logits 최대 차이: {worst_logits:.2e}")
    print(f"단계 칸 불일치: {phase_mismatch} / {T * N} step")
    print(f"student actions 최대 차이:   {worst_actions:.2e}")
    print(f"(참고) 기록 중 계단 env 단계 정답률: {agree:.1%}")
    ok = worst_logits < TOL and worst_actions < TOL and phase_mismatch == 0
    print("통과" if ok else "실패")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
