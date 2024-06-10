import torch.nn as nn
import os
from environment.rocket_lander import RocketLander
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


CONTINUE_TRAIN = True
MODEL_FILENAME = "VideoTraining"


if __name__ == "__main__":
    env_kwargs = {
       "render_mode" : "None",
    }

    env = make_vec_env(RocketLander, n_envs=8, vec_env_cls=DummyVecEnv, env_kwargs=env_kwargs)

    policy_kwargs = {
        "net_arch" : dict(pi=[64, 64], vf=[64,64]),
        "activation_fn" : nn.ReLU,
        #"lr_schedule": 1e-4,
    }

    if CONTINUE_TRAIN:
        if os.path.exists(MODEL_FILENAME + ".zip"):
            model = PPO.load(MODEL_FILENAME, env=env)
        else:
            raise FileNotFoundError
    else:
        model = PPO("MlpPolicy", env, device="cpu", policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=250*2000, progress_bar=True)
    model.save(MODEL_FILENAME)
