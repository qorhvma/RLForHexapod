import torch
from isaaclab.managers import SceneEntityCfg

def feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    # 이번 스텝에 "막 착지한" 발 (True/False)
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    # 착지하기 직전에 공중에 있던 시간
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    # threshold보다 오래 떠 있다가 착지하면 +, 짧으면 −
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # 서 있으라는 명령(속도 ≈ 0)일 때는 보상 없음
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward
