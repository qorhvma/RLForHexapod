from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

from isaaclab.envs import (
    ManagerBasedRLEnvCfg,
    VideoRecorderCfg,  # noqa: F401 -- used by the lesson 04 replacement
)
import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonManager

from RLforHexapod.assets import ROBOT_USDA

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
    spawn=sim_utils.UsdFileCfg(usd_path=str(ROBOT_USDA), scale=(0.1,0.1,0.1)),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.2),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*"],   # 18개 관절 전부
            stiffness=None,
            damping=None,
        ),
    },
)

@configclass
class ActionsCfg:
    # 모든 관절: 초기 자세 + 액션 × 0.5 rad 를 목표 각도로
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # 리셋 때 로봇을 init_state(초기 자세)로 되돌림
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")


@configclass
class RewardsCfg:
    alive = RewTerm(func=mdp.is_alive, weight=1.0)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

@configclass
class SceneCfg(InteractiveSceneCfg):
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=300, color=(0.0, 0.0, 0.0)))
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    robot = HEXAPOD_CFG


@configclass
class HexapodEnvCfg(ManagerBasedRLEnvCfg):
    """Step and reset the scene before adding teacher rewards."""

    scene: InteractiveSceneCfg = SceneCfg(num_envs=4, env_spacing=1.0)
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventsCfg = EventsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.sim.dt = 1.0 / 120.0
        self.episode_length_s = 20.0
        self.sim.physics = NewtonCfg(solver_cfg=MJWarpSolverCfg())
    

if __name__ == "__main__":
    sim = sim_utils.SimulationContext(
        sim_utils.SimulationCfg(dt=1 / 120, physics=NewtonCfg(solver_cfg=MJWarpSolverCfg()))
    )
