import gymnasium as gym
from gym.utils.play import PlayableGame, play
import numpy as np  # For handling observations and actions as arrays
from lunar_lander import LunarLander

env = LunarLander(render_mode="rgb_array")
env.reset()  # Reset the environment

keys_to_action = {
  # Left main engine thrust
  (ord('a'),): 1,  # 'a' key pressed for right thrust
  # Right main engine thrust
  (ord('d'),): 3,   # 'd' key pressed for left thrust
  # Main engine off
  (ord('w'),): 2,   # 'w' key pressed for upward thrust
  # Left landing leg deploy
  (ord('s'),): 0,     # 's' key pressed for no thrust
}

game = PlayableGame(env, keys_to_action=keys_to_action)
play(env, fps=30, keys_to_action=keys_to_action)