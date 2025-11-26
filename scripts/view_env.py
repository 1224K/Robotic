# run_env.py
import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from Robotic.tasks.direct.robotic.robotic_env import RoboticEnv
from Robotic.tasks.direct.robotic.robotic_env_cfg import RoboticEnvCfg


cfg = RoboticEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
env = RoboticEnv(cfg, render_mode="human")

num_envs = env.num_envs

# 真正的互動 loop
while simulation_app.is_running():
    obs, _ = env.reset()
    done = False
    cnt=0
    while simulation_app.is_running() and not done:
        actions = []
        for _ in range(num_envs):
            a = env.single_action_space.sample()  # numpy array shape (8,)
            # print(a)
            # if cnt<10:
            #     a = [1] * 8
            # else:
            #     a = [-1] * 8
            # cnt+=1
            # cnt %= 20
            actions.append(a)
        actions = torch.tensor(actions, device=env.device, dtype=torch.float32)
        obs, reward, terminated, truncated, info = env.step(actions)
        done = bool(terminated[0] or truncated[0])
        

simulation_app.close()
