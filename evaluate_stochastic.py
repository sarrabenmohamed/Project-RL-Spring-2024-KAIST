import torch.nn as nn

from stable_baselines3 import A2C, PPO
from environment.rocket_lander_stochastic import RocketLander

from sb3_contrib import RecurrentPPO












if __name__ == "__main__":
    env = RocketLander(render_mode="human")
    model = PPO.load("./NO_RANDOM_PPO/Final.zip")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        if term or trunc:
            break
       # env.render("human")