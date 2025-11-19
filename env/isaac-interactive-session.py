import argparse
from omni.isaac.lab.app import AppLauncher

# reference from IsaacLab/source/standalone/tutorials/02_scene/create_scene.py

help = """
run using isaaclab launcher
.$HOME/Workspace/nvidia/IsaacLab/isaaclab.sh -p ./isaac.py --argX
"""

# invoke isaac omniverse to launch sim
parser = argparse.ArgumentParser(description="Tutorial on using the interactive scene interface.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch simulation as omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import ArticulationCfg, AssetBaseCfg
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils import configclass

##
# Pre-defined configs
##
from omni.isaac.lab_assets import CARTPOLE_CFG, ANT_CFG, HUMANOID_CFG, SHADOW_HAND_CFG  # isort:skip
# todo: SPOT_CFG, FRANKA_CFG
@configclass
class SceneConfig(InteractiveSceneCfg):
	# setup scene with ground plane and lights
	ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
	dome_light = AssetBaseCfg(
			prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
	)
	#
	#cartpole: ArticulationCfg = CARTPOLE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
	#humanoid: ArticulationCfg = HUMANOID_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
	shadow_hand: ArticulationCfg = SHADOW_HAND_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

num_envs = 20
noise_amp = 0.2
action_amp = 5.0
is_headless = True

def reset_task(scene, robot):
	# reset coordinates of robots in all envs
	root_state = robot.data.default_root_state.clone()
	root_state[:,  :3] += scene.env_origins
	robot.write_root_state_to_sim(root_state)

	# reset configuration of all robots
	q_pos, q_vel = robot.data.default_joint_pos.clone(), robot.data.default_joint_vel.clone()
	q_pos += noise_amp * torch.rand_like(q_pos)
	robot.write_joint_state_to_sim(q_pos, q_vel)

	# reset scene
	scene.reset()


def step_task(sim, scene, robot, action):
	robot.set_joint_effort_target(action)
	scene.write_data_to_sim()
	sim.step()

if __name__ == '__main__':
	# setup simulator and scene
	sim_config = sim_utils.SimulationCfg(device='cuda')
	scene_config = SceneConfig(num_envs=num_envs, env_spacing=2.0)
	sim = SimulationContext(sim_config)
	sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])
	scene = InteractiveScene(scene_config)
	#robot = scene['humanoid']
	robot = scene['shadow_hand']
	#robot = scene['cartpole']
	sim_dt = sim.get_physics_dt()

	# call after defining sim & scene configs
	sim.reset()

	reset_task(scene, robot)
	for t in range(2000000):
		if not simulation_app.is_running():
			break
		#random_actions = 2.0 * torch.randn_like(robot.data.joint_pos) - 1.0
		#random_actions *= action_amp
		random_actions = torch.linspace(-1, 1, len(robot.data.joint_pos)) + torch.randn_like(robot.data.joint_pos)
		random_actions = action_amp * torch.sin(random_actions)

		step_task(sim, scene, robot, random_actions)
		scene.update(sim_dt)

	# close sim app
	simulation_app.close()