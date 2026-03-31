from hexapod.assets.robots.hexapod import HEXAPOD_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.locomotion.locomotion_env import LocomotionEnv


@configclass
class HexapodWalkEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10.0
    decimation = 2
    # - spaces definition
    action_scale = 0.5
    action_space = 18
    observation_space = 66
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=5.0, replicate_physics=True, clone_in_fabric=False
    )

    # robot
    robot: ArticulationCfg = HEXAPOD_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    joint_gears: list = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
    
    heading_weight: float = 0.5
    up_weight: float = 0.1

    energy_cost_scale: float = 0.05
    actions_cost_scale: float = 0.01
    alive_reward_scale: float = 2.0
    dof_vel_scale: float = 0.1

    death_cost: float = -1.0
    termination_height: float = -0.4

    angular_velocity_scale: float = 0.25
    contact_force_scale: float = 0.1


class HexapodWalkEnv(LocomotionEnv):
    cfg: HexapodWalkEnvCfg

    def __init__(self, cfg: HexapodWalkEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)