import torch.nn as nn
import os 
from lunar_lander import LunarLander
from rocket_lander import RocketLander
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv





CONTINUE_TRAIN = False
MODEL_FILENAME = "test"





if __name__ == "__main__":
    env_kwargs = {
       # "render_mode" : "human",
    }

    env = make_vec_env(RocketLander, n_envs=4, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    policy_kwargs = {
        "net_arch" : dict(pi=[64, 64], vf=[64,64]),
        "activation_fn" : nn.ReLU,
        #"lr_schedule": 1e-4,
    }

    if CONTINUE_TRAIN:
        if os.path.exists(MODEL_FILENAME + ".zip"):
            model = A2C.load(MODEL_FILENAME, env=env)
        else:
            raise FileNotFoundError
    else:
        model = A2C("MlpPolicy", env, device="cpu", policy_kwargs=policy_kwargs)

    model.learn(total_timesteps=25000)
    model.save(MODEL_FILENAME)
