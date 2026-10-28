# scripts/convert_urdf.py
from pathlib import Path
from isaacsim import SimulationApp

# instantiate the SimulationApp helper class
simulation_app = SimulationApp({"headless": False})
from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

ASSETS = Path(__file__).resolve().parents[1] / "src" / "RLforHexapod" / "assets"

cfg = UrdfConverterCfg(
    asset_path=str(ASSETS / "urdf" / "robot.urdf"),
    usd_dir=str(ASSETS / "robot"),
    usd_file_name="robot.usda",
    force_usd_conversion=True,
    make_instanceable=True,
    fix_base=False,            # 걷는 로봇이라 몸체 고정 X
    merge_fixed_joints=True,
    self_collision=False,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=1.0),
    ),
)

if __name__ == "__main__":
    converter = UrdfConverter(cfg)
    print("USD:", converter.usd_path)