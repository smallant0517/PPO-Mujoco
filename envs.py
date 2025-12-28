import numpy as np
import gymnasium as gym

def make_env(env_id: str, seed: int, idx: int):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed + idx)
        env.action_space.seed(seed + idx)
        env.observation_space.seed(seed + idx)
        return env
    return thunk

def make_vec_env(env_id: str, seed: int, num_envs: int):
    # MuJoCo 建議用 SyncVectorEnv 穩定
    envs = gym.vector.SyncVectorEnv([make_env(env_id, seed, i) for i in range(num_envs)])
    return envs

def get_space_info(envs):
    obs_shape = envs.single_observation_space.shape
    act_shape = envs.single_action_space.shape
    assert len(act_shape) == 1, "MuJoCo continuous action should be 1D vector."
    action_low = envs.single_action_space.low
    action_high = envs.single_action_space.high
    return obs_shape, act_shape[0], action_low, action_high