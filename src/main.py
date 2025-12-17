from stable_baselines3 import PPO
from cartpole_env import Cartpole
from stable_baselines3.common.env_util import make_vec_env

esp_ip = "192.168.1.50" 
env = make_vec_env(lambda: Cartpole(esp_ip, target_angles=(1.0, -0.5)), n_envs=1)

policy_kwargs = dict(net_arch=[64, 64])

model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs,
            learning_rate=3e-4, n_steps=256, batch_size=64, gamma=0.99)

model.learn(total_timesteps=50000)
model.save("PPO_UP_UP")
