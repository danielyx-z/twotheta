import os
import time
import numpy as np
from sbx import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "/dev/ttyUSB0"
BAUD = 921600
MODEL_NAME = "droq_pendulum_sbx"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints"
TOTAL_TIMESTEPS = 500000
STEPS_PER_SAVE = 4000

# Global gatekeeper to stop double-reset and fake steps during JAX JIT
RESET_COUNT = 0

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=2000, enable_viz=False))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    
    valid_files = []
    for f in files:
        try:
            num_part = "".join(filter(str.isdigit, f.split("_")[-1]))
            if num_part:
                valid_files.append((int(num_part), f))
        except ValueError:
            continue
            
    if not valid_files:
        return None
        
    valid_files.sort(key=lambda x: x[0])
    return os.path.join(CKPT_DIR, valid_files[-1][1])

def train():
    global RESET_COUNT
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        net_arch=[128, 128],
        dropout_rate=0.01,    # Standard DroQ regularization
        layer_norm=True       # Required to stabilize high UTD
    )


    params = {
        "learning_rate": 3e-4,
        "buffer_size": 100000, 
        "learning_starts": 1000, 
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
        "train_freq": (1, "step"),
        "gradient_steps": 20,
        "tensorboard_log": LOG_DIR
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PER_SAVE,
        save_path=CKPT_DIR,
        name_prefix=MODEL_NAME,
        save_replay_buffer=True
    )

    ckpt = latest_checkpoint()
    
    if ckpt:
        print(f"Loading Checkpoint: {ckpt}")
        model = SAC.load(ckpt, env=env)
        model.gradient_steps = 20

        try:
            start_steps = int("".join(filter(str.isdigit, ckpt.split("_")[-1])))
        except:
            start_steps = 0
    else:
        print("Starting DroQ from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params
        )
        start_steps = 0

    try:
        print("Begin training. (Wait for JAX compilation...)")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS - start_steps, 
            callback=checkpoint_callback,
            reset_num_timesteps=(start_steps == 0)
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train()