import torch.nn as nn
from stable_baselines3 import A2C
from environment.rocket_lander import RocketLander


if __name__ == "__main__":
    env = RocketLander(render_mode="human")
    model = A2C.load("test")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        if term or trunc:
            break
       # env.render("human")