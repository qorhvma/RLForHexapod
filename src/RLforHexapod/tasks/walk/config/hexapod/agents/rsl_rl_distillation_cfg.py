"""계단 teacher를 흉내 내는 student 증류 설정."""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
)


@configclass
class HexapodStudentRunnerCfg(RslRlDistillationRunnerCfg):
    seed = 42
    num_steps_per_env = 128
    max_iterations = 1500
    save_interval = 50
    experiment_name = "hexapod_stair_student"
    run_name = "A"
    obs_groups = {"student": ["student"], "teacher": ["policy"]}
    clip_actions = 2.0  # 계단 teacher 학습과 같게
    # student는 단계 정보를 받지 않으므로 기억으로 추정해야 한다
    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=True,
        # 행동 탐색 노이즈. 손실에 쓰이지 않아 학습되지 않는다 (teacher std 하한과 같은 값)
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.05),
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    # HexapodStatePPORunnerCfg.actor와 구조가 같아야 체크포인트를 불러올 수 있다
    teacher = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=64,  # 1.28초. 단계가 수 초씩 이어져서 튜토리얼 값(15)보다 길게 잡음
        max_grad_norm=1.0,
    )


@configclass
class HexapodStudentBRunnerCfg(HexapodStudentRunnerCfg):
    run_name = "B"


@configclass
class HexapodStudentAPhaseRunnerCfg(HexapodStudentRunnerCfg):
    run_name = "A_phase"


@configclass
class HexapodStudentATopEndRunnerCfg(HexapodStudentRunnerCfg):
    run_name = "A_top_end"


# 기억 시험: A-TopEnd에서 기억을 학습하는 구간만 늘린다. 제어 주기 0.02초 기준
@configclass
class HexapodStudentMem128RunnerCfg(HexapodStudentATopEndRunnerCfg):
    run_name = "A_top_end_mem128"
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=128,  # 2.56초
        max_grad_norm=1.0,
    )


@configclass
class HexapodStudentMem256RunnerCfg(HexapodStudentATopEndRunnerCfg):
    run_name = "A_top_end_mem256"
    num_steps_per_env = 256  # gradient_length를 나눠떨어지게 맞춘다
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=256,  # 5.12초
        max_grad_norm=1.0,
    )


# ===== 걷기·계단 통합 student =====
@configclass
class DualTeacherModelCfg(RslRlMLPModelCfg):
    class_name: str = "RLforHexapod.tasks.walk.config.hexapod.agents.dual_teacher:DualTeacherModel"
    walk_clip: float = 1.0   # HexapodStatePPORunnerCfg.clip_actions
    stair_clip: float = 2.0  # HexapodStairPPORunnerCfg.clip_actions


@configclass
class HexapodMixedStudentRunnerCfg(HexapodStudentATopEndRunnerCfg):
    experiment_name = "hexapod_student"
    run_name = "A_mixed"
    # teacher 순서는 DualTeacherModel이 정한 대로. policy 그룹이 계단 teacher 관측이다
    obs_groups = {"student": ["student"], "teacher": ["walk_teacher", "policy", "teacher_selector"]}
    # 두 teacher의 actor와 구조가 같아야 체크포인트를 불러올 수 있다
    teacher = DualTeacherModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.2, std_type="log", std_range=(0.05, 0.3)),
    )


@configclass
class HexapodMixedStudentBRunnerCfg(HexapodMixedStudentRunnerCfg):
    run_name = "B_mixed"


@configclass
class HexapodWalkOnlyStudentRunnerCfg(HexapodMixedStudentRunnerCfg):
    run_name = "A_walk_only"


@configclass
class HexapodStairOnlyStudentRunnerCfg(HexapodMixedStudentRunnerCfg):
    run_name = "A_stair_only"
