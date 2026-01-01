import os
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "COM8"
BAUD = 921600
MODEL_NAME = "single_pendulum_ppo"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints_ppo"
TOTAL_TIMESTEPS = 100000
STEPS_PER_SAVE = 3000

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=2000, enable_viz=False))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CKPT_DIR, files[-1])

def train():
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[64, 64]  # PPO typically uses a single shared network
    )

    params = {
        "learning_rate": 2e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "gae_lambda": 0.95,
        "gamma": 0.99,
        "clip_range": 0.2,
        "ent_coef": 0.0,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "tensorboard_log": LOG_DIR,
        "verbose": 1,
        "policy_kwargs": policy_kwargs,
        
    }

    ckpt = latest_checkpoint()
    
    if ckpt:
        print("Loading from latest checkpoint:", ckpt)
        model = PPO.load(ckpt, env=env, device="cpu",)
        start_steps = int(ckpt.split("_")[-1].split(".")[0])
    else:
        print("Starting PPO from scratch")
        model = PPO("MlpPolicy", env, device="cpu", **params)
        start_steps = 0

    total_steps = start_steps

    try:
        while total_steps < TOTAL_TIMESTEPS:
            model.learn(total_timesteps=STEPS_PER_SAVE, reset_num_timesteps=False)
            total_steps += STEPS_PER_SAVE

            path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_{total_steps}.zip")
            model.save(path)
            print(f"Saved checkpoint: {path}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    train()
