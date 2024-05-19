import torch.nn as nn
from lunar_lander import LunarLander
from stable_baselines3 import A2C














if __name__ == "__main__":
    env = LunarLander(render_mode="human")
    model = A2C.load("test")

    obs, info = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, term, trunc, info = env.step(action)
        if term or trunc:
            break
       # env.render("human")