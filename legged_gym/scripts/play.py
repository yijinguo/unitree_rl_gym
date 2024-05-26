import sys
sys.path.append("/mnt/pool1/sharehome/guoyijin/homework/unitree_rl_gym/")
from legged_gym import LEGGED_GYM_ROOT_DIR
import os

import isaacgym
from legged_gym.envs import *
from legged_gym.utils import  get_args, export_policy_as_jit, task_registry, Logger

import numpy as np
import torch


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 100)
    # env_cfg.terrain.num_rows = 10
    # env_cfg.terrain.num_cols = 1
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False

    # # env_cfg.terrain.terrain_proportions = [1.0, 0, 0, 0, 0] #smooth slope
    # # env_cfg.terrain.terrain_proportions = [0, 1.0, 0, 0, 0] #rough slope
    # env_cfg.terrain.terrain_proportions = [0, 0, 1.0, 0, 0] #stairs up
    # # env_cfg.terrain.terrain_proportions = [0, 0, 0, 1.0, 0] #stairs down
    # # env_cfg.terrain.terrain_proportions = [0, 0, 0, 0, 1.0] #discrete

    # env_cfg.commands.ranges.lin_vel_x = [0.5, 0.5]
    # env_cfg.commands.ranges.lin_vel_y = [0, 0]
    # env_cfg.commands.ranges.heading = [0, 0]

    env_cfg.env.test = True

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    video_path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
    for i in range(10*int(env.max_episode_length)):
        print(f"Epoch: {i}")
        actions = policy(obs.detach())
        obs, _, rews, dones, infos = env.step(actions.detach())
        # image_path = os.path.join(path, f'{i}.png')
        # env.save_RGB_image(image_path)
        env.get_frame()
        if i == 200:
            env.save_video(video_path)
            break
        

if __name__ == '__main__':
    EXPORT_POLICY = True
    RECORD_FRAMES = True
    MOVE_CAMERA = False
    args = get_args()
    play(args)
