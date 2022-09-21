from stable_baselines3 import PPO
import assistive_gym
import gym
import torch as T


env = gym.make('ScratchItchJaco-v1')

policy_kwargs = dict(ortho_init=False, net_arch=[dict(pi=[64, 64], vf=[64, 64])])

model = PPO(
    policy_kwargs=policy_kwargs,
    policy='MlpPolicy', 
    env=env, 
    verbose=1,
    clip_range=0.2,
    ent_coef=1e-3,
    gamma=0.95,
    gae_lambda=0.99,
    n_epochs=10,
    learning_rate=0.0003,
    batch_size=64,
    n_steps=2048,
    max_grad_norm=0.5,
    seed=200,
    )

model.learn(total_timesteps=1000000)

model.save("models/ppo_baseline")
