import math

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg, 
)
import RLforHexapod.tasks.walk.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager
from isaaclab_newton.sim.schemas import NewtonArticulationCfg
from isaaclab.sensors import ContactSensorCfg

from RLforHexapod.assets import ROBOT_USDA, OBSTACLES_DIR

JOINTS = [
    "L1_tibia", "L1_femur", "L1_coxa", 
    "L2_tibia", "L2_femur", "L2_coxa", 
    "L3_tibia", "L3_femur", "L3_coxa",
    "L4_tibia", "L4_femur", "L4_coxa", 
    "L5_tibia", "L5_femur", "L5_coxa", 
    "L6_tibia", "L6_femur", "L6_coxa"
]

HEXAPOD_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USDA)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.3),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            ".*_coxa": 0.0,
            ".*_femur": math.radians(20.0),  # 0=아래로 수직, 90=수평 (한계 0~180)
            ".*_tibia": math.radians(-110.0),  # -90=femur와 일직선, 값이 작을수록 아래로 굽힘 (한계 -250~-70: 아래 160, 위 20)
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # MJWarp는 관절 속도 한계를 무시하므로 토크-속도 곡선으로 속도를 제한한다.
        # coxa=MG996R 6V 데이터시트, femur/tibia=RC920DMG 임시값(사양 불명, 7V 실측으로 교체)
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],   # 18개 관절 전부
            # None이면 Newton 액추에이터의 kp/kd가 0이 된다.
            stiffness=100.0,
            damping=1.0,
            # 감속기 반사 관성(MG996R 급 ~0.005-0.01). 0이면 가벼운 tibia가 불안정
            armature=0.005,
            saturation_effort={".*_coxa": 1.08, ".*_(femur|tibia)": 1.86},  # 스톨 토크 [N·m]
            actuator_velocity_limit={".*_coxa": 7.48, ".*_(femur|tibia)": 6.5},  # 무부하 속도 [rad/s]
            actuator_effort_limit={".*_coxa": 0.75, ".*_(femur|tibia)": 1.3},  # 지속 토크, 스톨의 70% [N·m]
        ),
    },
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000, color=(0.75, 0.75, 0.75)))
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = HEXAPOD_CFG
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )


@configclass
class ObservationsCfg:
    # ObservationsCfg 안에는 ObsGroup만이 들어갈 수 있다고 한다.
    # 같은 task 내에서 ObsGroup을 바꿔 껴서 쉽게 할 수 있도록 했다.
    # 사용 예시로 증류가 있다.
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel) 
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_height"})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    # 센서로 얻을 수 있는 값.
    

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], use_default_offset=True,
        scale={".*_coxa":0.5, ".*_femur": 1.0, ".*_tibia": 1.0},    
        # scale 증가 필요
    )


@configclass
class EventsCfg:
    # 실제 서보 회전 범위. robot.usda(LFS)의 0~270 override를 시작 시 덮어씀, coxa는 USD의 ±90 그대로
    joint_limits = EventTerm(func=mdp.set_joint_pos_limits, mode="startup", params={
        "limits_deg": {".*_femur": (0.0, 180.0), ".*_tibia": (-250.0, -70.0)}})
    # 리셋 때 로봇을 init_state(초기 자세)로 되돌림
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=0.01)
    # 보상 = exp(-오차² / std²). std는 명령 크기에 맞춘다: 0.2m/s 명령에 서 있으면 0.02, std 0.5였을 때는 0.85
    track_lin_vel = RewTerm(func=mdp.track_lin_vel_xy_exp, params={"command_name": "base_velocity", "std": 0.1}, weight=1)
    # 0.5rad/s 명령에 회전하지 않으면 0.02
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_exp, params={"command_name": "base_velocity", "std": 0.25}, weight=0.5)
    # 명령 범위 0.07m에 맞춘 std: 4cm 어긋나면 0.02
    track_base_height = RewTerm(func=mdp.track_base_height_exp, params={"command_name": "base_height", "std": 0.02}, weight=1.0)
    # 땅에 닿은 다리의 tibia 수직도 평균. std 30°: 10° 기울면 0.90, 20° 0.64, 30° 0.37, 45° 0.11
    # cos는 30°에도 0.87이라 자세와 상관없이 받는 점수가 되어 수직을 가르치지 못했다
    tibia_vertical = RewTerm(func=mdp.tibia_vertical, weight=0.5, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=mdp.FOOT_BODIES, preserve_order=True),
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.FOOT_BODIES, preserve_order=True),
        "std": math.radians(30.0)})
    # 발을 든 시간이 0.3초에서 벗어나면 착지마다 벌점. 0.1초 어긋나면 0.63 × weight. 초당 10번 착지가 전부 크게 어긋나면 -0.5/s
    feet_air_time_error = RewTerm(func=mdp.feet_air_time_error, weight=-0.05, params={
        "command_name": "base_velocity", "target": 0.3, "std": 0.1,
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.FOOT_BODIES, preserve_order=True)})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-7)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.05})
    bad_ori  = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})  # 약 57°
    sim_nan = DoneTerm(func=mdp.sim_nan)


@configclass
class CommandCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        # True면 모든 env에서 ang_vel_z가 heading 오차로 덮어써져 회전 명령이 무시된다 (rel_heading_envs 기본값 1.0)
        heading_command=False,
        rel_standing_envs=0.1,  # 10%는 정지 명령: 계단 앞이나 꼭대기에서 멈추는 동작에 필요
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            # 공중·지지 각 0.3초인 삼각 보행에서 0.2m/s는 보폭 6cm. 0.5m/s는 15cm라 coxa 보폭(9~12.6cm)을 넘는다
            lin_vel_x=(-0.2, 0.2),
            lin_vel_y=(-0.2, 0.2),  # 옆걸음: 계단 앞에서 좌우 위치 맞추기
            ang_vel_z=(-0.5, 0.5),  # 계단 앞에서 방향 맞추기
        ),
        debug_vis=True,
    )
    # 몸통 원점 높이. 0.24m: femur 약 30° (다리를 거의 편 0.254m에서는 발을 들 여유가 없다), 0.17m: femur 약 80°
    # 지금 기본 자세(femur 20°, tibia -110°)는 0.248m
    base_height = mdp.BaseHeightCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        height_range=(0.17, 0.24),
    )


@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """Step and reset the scene before adding teacher rewards."""

    # 7대장 (7개의 메니저가 있고, 그 메니저들의 설정이다.)
    scene: InteractiveSceneCfg = SceneCfg(num_envs=4, env_spacing=1.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    commands: CommandCfg = CommandCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 200.0  # 정책 50Hz (decimation 4)
        self.episode_length_s = 20.0
        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg(nconmax=100, njmax=256), use_cuda_graph=True)
        # Isaac Sim 대비 환경 설정이 간편함, 병렬 학습 속도는 비슷 비슷, 오픈소스라 마개조 가능, 다만 신버전이라 지원이 부족할 수 있음.
    

# Stairs-cfg  =====================================================

# 계단 형상: 씬 배치(StairsSceneCfg)와 특권 관측이 같은 값을 쓰도록 한 곳에서 정의
STAIRS = mdp.StairsGeometryCfg(x0=0.5, step_height=0.16)

@configclass
class StairsSceneCfg(SceneCfg):
    # 자기충돌을 켠다: 계단에서는 다리끼리 가까워져, 꺼져 있으면 서로 통과하는 동작을 배운다
    robot = HEXAPOD_CFG.replace(spawn=HEXAPOD_CFG.spawn.replace(
        articulation_props=[NewtonArticulationCfg(self_collision_enabled=True)]))
    # 다리끼리의 접촉만 재는 센서: femur·tibia 바디 쌍별 힘 (leg_self_contacts 벌점이 읽는다)
    leg_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*(femur|tibia).*",
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*(femur|tibia).*"],
        history_length=3,
    )
    stairs = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Stairs",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(OBSTACLES_DIR / f"stairs_{round(STAIRS.step_height * 1000)}mm.usda")),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(STAIRS.x0, 0.0, 0.0)),  # 로봇 앞 0.5m
    )


@configclass
class StairsTerminationsCfg(TerminationsCfg):
    # 조금 기울어지는 건 괜찮다. 약 29° 넘으면 실패
    bad_ori = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.5})
    # 몸통 중심이 계단 폭(1.0m, stairs_*.usda의 y scale) 밖으로 나가면 실패
    off_stairs = DoneTerm(func=mdp.out_of_stairs_width, params={"half_width": STAIRS.width/2})
    sim_blowup = DoneTerm(func=mdp.joint_vel_out_of_manual_limit, params={"max_velocity": 50.0})
    # 꼭대기에 올라 멈추면 성공
    success = DoneTerm(func=mdp.stair_success, time_out=True)


@configclass
class StairsEventsCfg(EventsCfg):
    # x_ranges 순서 = 단계 번호 (몸통 x 범위). 0번 전진: 수직면까지 멀리서, 1번 앞발 옮기기: 앞발이 수직면 3~5cm 앞
    stair_reset = EventTerm(func=mdp.reset_stair_state, mode="reset",
                            params={"x_ranges": ((-0.1, 0.1), (0.30, 0.32)), "buffer_ratio": 0.5})


@configclass
class StairsRewardsCfg:
    # 단계를 마칠 때마다 1번 (weight = 점수)
    phase_bonus = RewTerm(func=mdp.stair_phase_bonus, weight=10.0)
    # 뒷발 앞 전진과 뒷발 옮기기에서 앞다리를 수직면과 평행하게 돌린 비율의 신기록 (weight = 다리 하나를 끝까지 돌린 점수)
    front_fold = RewTerm(func=mdp.stair_front_fold, weight=5.0)
    # 옮기는 발이 단 위에 처음 놓일 때, 발마다 단계당 1번 (weight = 발 하나의 점수)
    foot_placed = RewTerm(func=mdp.stair_foot_placed, weight=5.0)
    # 옮기는 발이 단 끝에라도 처음 닿을 때, 발마다 단계당 1번 (weight = 발 하나의 점수)
    edge_touch = RewTerm(func=mdp.stair_edge_touch, weight=2.0)
    # 지금 단계의 신기록 갱신량. 전진: 몸통 x, 옮기기: 발끝 높이 + 몸통 x (weight = 1m당 점수)
    progress = RewTerm(func=mdp.stair_progress, weight=30.0)
    # 모든 단계에서 여섯 다리의 tibia가 지면에 수직일수록 (weight = 초당 최대 점수)
    tibia_vertical = RewTerm(func=mdp.stair_tibia_vertical, weight=0.5)
    # 옮기기 중 제자리에서 떨어진 나머지 발의 수 (weight = 발 하나당 초당 벌점)
    standing_feet_off = RewTerm(func=mdp.stair_standing_feet_off, weight=-1.0)
    # 모든 단계에서 몸통의 후진 속도 (weight = 후진 1m당 점수)
    backward = RewTerm(func=mdp.stair_backward, weight=-30.0)
    # 꼭대기에서 멈출수록 (weight = 초당 최대 점수)
    settle_still = RewTerm(func=mdp.stair_settle_still, weight=4.0)
    # 실패: weight × step_dt(0.02)가 실제 점수 → -10점
    termination_penalty = RewTerm(func=mdp.is_terminated_term, weight=-500.0, params={
        "term_keys": ["base_low", "bad_ori", "off_stairs", "sim_blowup", "sim_nan"]})  # 성공 제외

    # 동작 규제
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)

    # 동작 규제
    undesired_contacts = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={
        "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["body", "femur.*", "coxa.*"]),
        "threshold": 1.0})
    # 서로 다른 다리끼리 닿은 쌍 하나당 초당 벌점. 너무 크면 발을 옮기는 시도 자체를 포기한다
    leg_self_contact = RewTerm(func=mdp.leg_self_contacts, weight=-2.0, params={
        "sensor_name": "leg_contacts", "threshold": 1.0})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.5)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-7)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1e-3)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)


@configclass
class StairsObservationsCfg(ObservationsCfg):
    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # 속도·높이 명령을 따르지 않으므로 뺀다
        command = None
        height_command = None
        # 계단 teacher 특권 관측
        # 1. 발끝이 지형에서 떠 있는 높이 (6) + 발끝 아래 지형의 몸통 대비 깊이 (6)
        feet_height = ObsTerm(func=mdp.feet_height_above_terrain, params={
            "asset_cfg": SceneEntityCfg("robot", body_names="tibia.*"), "stairs": STAIRS})
        terrain_under_feet = ObsTerm(func=mdp.terrain_height_under_feet, params={
            "asset_cfg": SceneEntityCfg("robot", body_names="tibia.*"), "stairs": STAIRS})
        # 2. 단 높이 (1)
        step_height = ObsTerm(func=mdp.stairs_step_height, params={"stairs": STAIRS})
        # 3. 지금 할 일: 단계 (7), 옮기는 발 위치와 tibia 수직도 (6), 전진 목표까지 거리와 방향 (3)
        phase = ObsTerm(func=mdp.stair_phase_obs)
        feet_target = ObsTerm(func=mdp.stair_feet_target_obs)
        body_target = ObsTerm(func=mdp.stair_body_target_obs)

    policy: PolicyCfg = PolicyCfg()


@configclass
class HexapodStairEnvCfg(HexapodEnvCfg):
    # ManagerBaseRLEnvCfg를 불러온 HexapodEnvCfg를 불러온 HexapodStairEvnCfg, 값만 바꾼거라 사실상 껍데기.
    scene: SceneCfg = StairsSceneCfg(num_envs=4, env_spacing=3.0)
    observations: ObservationsCfg = StairsObservationsCfg()
    events: EventsCfg = StairsEventsCfg()
    rewards: RewardsCfg = StairsRewardsCfg()
    terminations: TerminationsCfg = StairsTerminationsCfg()
    # StairTracker가 읽는 값
    stairs: mdp.StairsGeometryCfg = STAIRS
    stair_gait: mdp.StairGaitCfg = mdp.StairGaitCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 45.0  # 단마다 6단계 × 3단 + 마지막 전진, 정지


# Student-cfg  ====================================================
# 관측 시험 계획: docs/student_observation_test.md

@configclass
class StudentAObservationsCfg(StairsObservationsCfg):
    """teacher는 기존 policy 그룹을 그대로 보고, student는 아래 student 그룹만 본다."""

    @configclass
    class StudentCfg(ObsGroup):
        # student_* 함수는 센서 오차 설정(sensor_error, mdp/sensor_model.py)이 없으면 원래 관측과 같다
        # 실제 관절각 (엔코더가 있다고 가정)
        joint_pos = ObsTerm(func=mdp.student_joint_pos)
        joint_vel = ObsTerm(func=mdp.student_joint_vel)  # sensor_error.commanded_joints가 아니면 joint_vel_rel과 같다
        # L2 내장 IMU로 측정 또는 계산
        base_ang_vel = ObsTerm(func=mdp.student_base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.student_projected_gravity)
        last_action = ObsTerm(func=mdp.last_action)
        feet_contact = ObsTerm(func=mdp.student_feet_contact, params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=mdp.FOOT_BODIES, preserve_order=True)})
        # 앞쪽 x 2cm 간격, 옆쪽 y 10cm 간격, 41 x 5 = 205칸
        height_map = ObsTerm(func=mdp.stairs_height_map, noise=GaussianNoiseCfg(std=0.01), params={
            "asset_cfg": SceneEntityCfg("robot"), "stairs": STAIRS,
            "x_range": (-0.24, 0.56), "y_range": (-0.2, 0.2), "spacing": (0.02, 0.1)})

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = True  # 이 값이 있어야 noise가 적용된다

    student: StudentCfg = StudentCfg()


@configclass
class HexapodStairStudentAEnvCfg(HexapodStairEnvCfg):
    observations: ObservationsCfg = StudentAObservationsCfg()
    # student 관측과 단계 계산의 센서 오차 (기본 = 오차 없음). 평가·학습 스크립트가 바꾼다
    sensor_error: mdp.SensorErrorCfg = mdp.SensorErrorCfg()

    def __post_init__(self):
        super().__post_init__()
        # 실제 운용처럼 기억이 빈 상태에서 계단 앞부터 시작. stair_success = 처음부터 끝까지 오른 비율
        self.events.stair_reset.params["buffer_ratio"] = 0.0


@configclass
class StudentBObservationsCfg(StudentAObservationsCfg):
    @configclass
    class StudentCfg(StudentAObservationsCfg.StudentCfg):
        # 서보 출력을 읽지 못하는 경우. 관절 정보는 last_action(명령값)뿐이다
        joint_pos = None
        joint_vel = None

    student: StudentCfg = StudentCfg()


@configclass
class HexapodStairStudentBEnvCfg(HexapodStairStudentAEnvCfg):
    observations: ObservationsCfg = StudentBObservationsCfg()


# 원인 분리 시험: A에서 한 가지만 바꾼다. 실물에 쓸 설정이 아니다.

@configclass
class StudentAPhaseObservationsCfg(StudentAObservationsCfg):
    @configclass
    class StudentCfg(StudentAObservationsCfg.StudentCfg):
        # 실물 센서(Point-LIO, 관절각, 발 접촉)로 계산한다고 가정한 단계 (7). 판정은 phase_estimator, 오차는 sensor_error
        phase = ObsTerm(func=mdp.stair_phase_est_obs)

    @configclass
    class PhaseNetCfg(ObsGroup):
        """단계 추정 신경망(agents/phase_net.py)만 보는 추가 입력. A-Phase 정책은 보지 않는다 (체크포인트 호환)."""

        # 라이다로 잰 몸통 x (첫 수직면 기준 + 수직면마다 지난 거리). 라이다 x 오차가 들어간다
        body_x = ObsTerm(func=mdp.lidar_body_x, params={"stairs": STAIRS})
        # IMU + 관절각 FK로 계산한 몸통 기준 발끝 높이 (6). 엔코더·IMU 기울기 오차가 들어간다
        feet_below_body = ObsTerm(func=mdp.feet_below_body)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class PhaseRuleCfg(ObsGroup):
        """단계 추정 신경망만 보는 규칙 추정기의 판정과 판정 값 (40). 신경망이 규칙을 출발점으로 고치게 한다."""

        rule = ObsTerm(func=mdp.stair_rule_features)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    @configclass
    class PhaseTimerCfg(ObsGroup):
        """단계 추정 신경망만 보는 규칙 추정기가 지금 단계에 머문 시간 (1). 잠깐 멈춘 것과 편향으로 굳은 것을 가른다.
        phase_rule과 따로 둔 그룹이라 이것 없이 학습한 신경망도 그대로 평가할 수 있다."""

        time = ObsTerm(func=mdp.stair_rule_phase_time)

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    student: StudentCfg = StudentCfg()
    phase_net: PhaseNetCfg = PhaseNetCfg()
    phase_rule: PhaseRuleCfg = PhaseRuleCfg()
    phase_timer: PhaseTimerCfg = PhaseTimerCfg()


@configclass
class HexapodStairStudentAPhaseEnvCfg(HexapodStairStudentAEnvCfg):
    observations: ObservationsCfg = StudentAPhaseObservationsCfg()
    # 센서 오차가 없으면 student가 참값과 같은 단계를 본다. 오차 평가는 scripts/eval_stairs.py에서 바꾼다
    phase_estimator: mdp.PhaseEstimatorCfg = mdp.PhaseEstimatorCfg()


@configclass
class HexapodStairStudentAPhaseNoiseEnvCfg(HexapodStairStudentAPhaseEnvCfg):
    # 센서 오차를 넣어 증류한다. 같은 오차가 단계 계산과 student 관측(높이맵, 관절각, 접촉)에 함께 들어간다
    # 관측 구성은 A-Phase와 같아 그 체크포인트에서 이어 학습할 수 있다
    # 실측값이 없어 오차 시험(--error_sweep)에서 성공률이 꺾이는 지점으로 잡은 최댓값. 에피소드마다 0 ~ 최댓값에서 뽑는다
    # 2026-11-12 이전 런(model_7600 등)은 오차가 단계 계산에만 들어간 옛 방식으로 학습됐다
    sensor_error: mdp.SensorErrorCfg = mdp.SensorErrorCfg(
        lidar_x_bias=0.015,
        lidar_x_noise=0.01,
        foot_height_bias=0.01,
        foot_height_noise=0.005,
        encoder_bias=math.radians(3.0),
        contact_flip=0.05,
        randomize=True,
    )
    phase_estimator: mdp.PhaseEstimatorCfg = mdp.PhaseEstimatorCfg(delay=0.1, randomize_delay=True)


@configclass
class StudentAPhaseAuxObservationsCfg(StudentAObservationsCfg):
    """student 관측은 A와 같다. 참값 단계를 학습 정답으로만 따로 둔다 (agents/phase_aux.py)."""

    @configclass
    class PhaseLabelCfg(ObsGroup):
        phase = ObsTerm(func=mdp.stair_phase_obs)  # teacher가 보는 단계와 같은 one-hot (7). 노이즈 없음

        def __post_init__(self):
            self.concatenate_terms = True
            self.enable_corruption = False

    phase_label: PhaseLabelCfg = PhaseLabelCfg()


@configclass
class HexapodStairStudentAPhaseAuxEnvCfg(HexapodStairStudentAEnvCfg):
    observations: ObservationsCfg = StudentAPhaseAuxObservationsCfg()


@configclass
class StudentTopEndTerminationsCfg(StairsTerminationsCfg):
    # 꼭대기에 서 있는 동안의 데이터가 학습을 방해하는지 확인한다
    reached_top = DoneTerm(func=mdp.stair_reached_top, time_out=True)


@configclass
class HexapodStairStudentATopEndEnvCfg(HexapodStairStudentAEnvCfg):
    terminations: TerminationsCfg = StudentTopEndTerminationsCfg()


# 걷기·계단 통합 student =============================================
# 모든 env에 계단이 있고, 걷기 env만 계단 뒤쪽 평지에서 시작한다.
# 걷기는 0.2m/s로 20초, 최대 4m 움직인다. 4m 뒤에서 시작하면 자기 계단(x +0.5부터)에 닿지 않고,
# 간격 10m면 뒤쪽 env의 계단(끝이 x -8.1)에도 닿지 않는다

WALK_START_X = -4.0
MIXED_ENV_SPACING = 10.0


@configclass
class MixedSceneCfg(StairsSceneCfg):
    leg_contacts = None  # 벌점용 센서. 증류는 보상을 쓰지 않는다


@configclass
class MixedEventsCfg(StairsEventsCfg):
    stair_reset = EventTerm(func=mdp.reset_mixed_state, mode="reset", params={
        "x_ranges": ((-0.1, 0.1), (0.30, 0.32)), "buffer_ratio": 0.0, "walk_start_x": WALK_START_X})


@configclass
class MixedRewardsCfg:
    # 증류는 보상을 쓰지 않는다. StairTracker 갱신과 계단 지표 기록을 위해 이 항목만 둔다
    phase_bonus = RewTerm(func=mdp.stair_phase_bonus, weight=1.0)


@configclass
class MixedTerminationsCfg(StudentTopEndTerminationsCfg):
    # 각 teacher를 학습할 때와 같은 조건으로 끝낸다
    bad_ori = DoneTerm(func=mdp.mixed_bad_orientation, params={"stair_limit": 0.5, "walk_limit": 1.0})
    off_stairs = DoneTerm(func=mdp.stair_out_of_width, params={"half_width": STAIRS.width/2})
    walk_time_out = DoneTerm(func=mdp.walk_time_out, params={"time_s": 20.0}, time_out=True)


@configclass
class MixedStudentObservationsCfg(StudentAObservationsCfg):
    """policy = 계단 teacher, walk_teacher = 걷기 teacher, student = 실물에서 얻을 수 있는 값."""

    @configclass
    class StudentCfg(StudentAObservationsCfg.StudentCfg):
        # 조종기 명령 (4). 계단 env에는 "전진하며 높게 서기"를 보여 준다
        command = ObsTerm(func=mdp.student_command, params={"stair_velocity": (0.2, 0.0, 0.0), "stair_height": 0.24})

    @configclass
    class TeacherSelectorCfg(ObsGroup):
        is_stair = ObsTerm(func=mdp.stair_env_flag)

        def __post_init__(self):
            self.concatenate_terms = True

    student: StudentCfg = StudentCfg()
    walk_teacher: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()
    teacher_selector: TeacherSelectorCfg = TeacherSelectorCfg()


@configclass
class MixedStudentBObservationsCfg(MixedStudentObservationsCfg):
    @configclass
    class StudentCfg(MixedStudentObservationsCfg.StudentCfg):
        # 서보 출력을 읽지 못하는 경우. 관절 정보는 last_action(명령값)뿐이다
        joint_pos = None
        joint_vel = None

    student: StudentCfg = StudentCfg()


@configclass
class HexapodMixedStudentEnvCfg(HexapodStairStudentAEnvCfg):
    scene: SceneCfg = MixedSceneCfg(num_envs=4, env_spacing=MIXED_ENV_SPACING)
    observations: ObservationsCfg = MixedStudentObservationsCfg()
    events: EventsCfg = MixedEventsCfg()
    rewards: RewardsCfg = MixedRewardsCfg()
    terminations: TerminationsCfg = MixedTerminationsCfg()
    stair_ratio: float = 0.5  # 계단 env 비율


@configclass
class HexapodMixedStudentBEnvCfg(HexapodMixedStudentEnvCfg):
    observations: ObservationsCfg = MixedStudentBObservationsCfg()


@configclass
class MixedStudentAPhaseObservationsCfg(MixedStudentObservationsCfg):
    """통합 student + 단계 입력 (A-Phase 구조). student 관측 순서: Student-A 관측, 명령 (4), 단계 (7).
    A-Phase와 비교하면 단계 칸 앞에 명령 4칸이 끼어 있다 (scripts/expand_student.py가 이 자리에 칸을 넣는다).
    걷기 env는 계단에 닿지 않으므로 단계가 늘 첫 전진(계단 앞으로 가기)이다."""

    @configclass
    class StudentCfg(MixedStudentObservationsCfg.StudentCfg):
        phase = ObsTerm(func=mdp.stair_phase_est_obs)

    student: StudentCfg = StudentCfg()
    # 단계 추정 신경망용 입력. 나중에 이 env에서 신경망을 다시 학습할 수 있게 A-Phase와 같은 그룹을 둔다
    phase_net: StudentAPhaseObservationsCfg.PhaseNetCfg = StudentAPhaseObservationsCfg.PhaseNetCfg()
    phase_rule: StudentAPhaseObservationsCfg.PhaseRuleCfg = StudentAPhaseObservationsCfg.PhaseRuleCfg()
    phase_timer: StudentAPhaseObservationsCfg.PhaseTimerCfg = StudentAPhaseObservationsCfg.PhaseTimerCfg()


@configclass
class HexapodMixedStudentAPhaseEnvCfg(HexapodMixedStudentEnvCfg):
    observations: ObservationsCfg = MixedStudentAPhaseObservationsCfg()
    # 오차가 없으면 student가 참값 단계를 본다 (A-Phase와 같음). 오차 평가는 scripts/eval_stairs.py에서 바꾼다
    phase_estimator: mdp.PhaseEstimatorCfg = mdp.PhaseEstimatorCfg()


@configclass
class HexapodMixedStudentAPhaseCmdEnvCfg(HexapodMixedStudentAPhaseEnvCfg):
    # 실물 서보는 관절각을 읽을 수 없다: 관절 관측·발끝·앞다리 접힘에 마지막 목표각, 관절 속도 자리에 목표각 변화율.
    # 관측 크기가 같아 Hexapod-Student-A-Phase 체크포인트에서 이어 학습한다. teacher는 참값을 본다
    sensor_error: mdp.SensorErrorCfg = mdp.SensorErrorCfg(commanded_joints=True)


# 확인용: 통합 학습 전에 한쪽씩만 증류한다
@configclass
class HexapodWalkOnlyStudentEnvCfg(HexapodMixedStudentEnvCfg):
    stair_ratio: float = 0.0


@configclass
class HexapodStairOnlyStudentEnvCfg(HexapodMixedStudentEnvCfg):
    stair_ratio: float = 1.0
