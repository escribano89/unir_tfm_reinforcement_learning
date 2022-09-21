from stable_baselines3 import TD3
import assistive_gym
import gym
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise

env = gym.make('ScratchItchJaco-v1')

n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

policy_kwargs = dict(net_arch=[400, 300])

model = TD3(
    policy_kwargs=policy_kwargs,
    policy='MlpPolicy', 
    action_noise=action_noise,
    env=env, 
    verbose=1,
    gamma=0.99,
    learning_rate=0.001,
    batch_size=100,
    seed=42,
    buffer_size=64000,
    policy_delay=2,
    tau=0.005,
    learning_starts=10000,
    target_noise_clip=0.5,
    target_policy_noise=0.2,
    train_freq=(2, "step") 
    )

model.learn(total_timesteps=1000000)

model.save("models/td3_baseline")
