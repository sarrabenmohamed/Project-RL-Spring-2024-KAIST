import torch.nn as nn
import os 

from environment.rocket_lander import RocketLander
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList





CONTINUE_TRAIN = True
MODEL_FILENAME = "Final_12"

CURRENT_RUN_FOLDER = "./5_percent_rand_1_percent_noise_PPO"






if __name__ == "__main__":
    if not os.path.exists(CURRENT_RUN_FOLDER):
        os.mkdir(CURRENT_RUN_FOLDER)
    env_kwargs = {
        "render_mode" : "None",
        "dynamics_rand" : True,
        "input_noise" : True,
        "dynamics_rand_deviation":0.05,
        "input_noise_deviation":0.01
    }

    env = make_vec_env(RocketLander, n_envs=16, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs, seed=2)

    eval_callback = EvalCallback(env, best_model_save_path=f"{CURRENT_RUN_FOLDER}/logs_eval_callback/",
                             log_path=f"{CURRENT_RUN_FOLDER}/logs_eval_callback/", eval_freq=100000,
                             deterministic=True, render=False)
    
    checkpoint_callback = CheckpointCallback(save_freq=100000, save_path=f"{CURRENT_RUN_FOLDER}/logs_checkpoint_callback/", name_prefix="rl_model", 
                                             save_replay_buffer=True, save_vecnormalize=True)
    policy_kwargs = {
        "net_arch" : dict(pi=[64, 64], vf=[64,64]),
        "activation_fn" : nn.ReLU,
        #"lstm_hidden_size" : 64,
        #"enable_critic_lstm" : False,

        #"lr_schedule": 1e-4,
    }

    if CONTINUE_TRAIN:
        if os.path.exists(f"{CURRENT_RUN_FOLDER}/{MODEL_FILENAME}.zip"):
            model = PPO.load(f"{CURRENT_RUN_FOLDER}/{MODEL_FILENAME}", env=env)
        else:
            raise FileNotFoundError
    else:
        model = PPO("MlpPolicy", env, device="cpu", policy_kwargs=policy_kwargs, n_steps=2048, verbose=1, tensorboard_log=f"{CURRENT_RUN_FOLDER}/tensorboard_log", learning_rate=1e-6, ent_coef=1e-4)


    for i in range(13, 1000):
        model.learn(total_timesteps=500001, progress_bar=True, callback=CallbackList([eval_callback, checkpoint_callback]))
        model.save(f"{CURRENT_RUN_FOLDER}/{MODEL_FILENAME}_{i}")
