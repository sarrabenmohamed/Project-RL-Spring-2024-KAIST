import torch.nn as nn
from lunar_lander import LunarLander
from stable_baselines3 import A2C, PPO
from rocket_lander import RocketLander

from sb3_contrib import RecurrentPPO












if __name__ == "__main__":
    env = RocketLander(render_mode="human")
    model = PPO.load("/home/maxfactor/Documents/KAIST/RL/Project-RL-Spring-2024-KAIST/5_percent_rand_1_percent_noise_PPO/Final_12_13.zip")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        if term or trunc:
            break
       # env.render("human")