import os
import pickle
import struct
import math
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "COM8"
BAUD = 921600
MODEL_NAME = "single_pendulum"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints"
TOTAL_TIMESTEPS = 600000
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
    # Sort by the step count at the end of the filename
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CKPT_DIR, files[-1])

def migrate_buffer(model, replay_path):
    """Manually moves transitions from an old buffer to a new one if sizes differ."""
    if not os.path.exists(replay_path):
        print("No replay buffer found to migrate.")
        return

    print(f"Loading and migrating replay buffer: {replay_path}")
    with open(replay_path, "rb") as f:
        try:
            old_buffer = pickle.load(f)
        except Exception as e:
            print(f"Failed to load buffer file: {e}")
            return

    # Get current buffer from model
    new_buffer = model.replay_buffer
    
    try:
        n_transitions = old_buffer.size()
        print(f"Migrating {n_transitions} transitions...")

        # Map data from old to new
        # We use min() to prevent overflow if the new buffer is actually smaller
        limit = min(n_transitions, new_buffer.buffer_size)
        
        new_buffer.observations[:limit] = old_buffer.observations[:limit]
        new_buffer.actions[:limit] = old_buffer.actions[:limit]
        new_buffer.rewards[:limit] = old_buffer.rewards[:limit]
        new_buffer.next_observations[:limit] = old_buffer.next_observations[:limit]
        new_buffer.dones[:limit] = old_buffer.dones[:limit]
        
        # Update internal pointers
        new_buffer.pos = limit % new_buffer.buffer_size
        new_buffer.full = limit >= new_buffer.buffer_size
        
        print(f"Successfully migrated to new buffer (Size: {new_buffer.buffer_size})")
    except Exception as e:
        print(f"Migration failed due to shape mismatch: {e}")
        print("Starting with an empty buffer instead.")

def train():
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        activation_fn=nn.Tanh, 
        net_arch=dict(pi=[64, 64], qf=[64, 64])
    )

    params = {
        "learning_rate": 2e-4,
        "buffer_size": 80000, 
        "learning_starts": 1000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto_0.1",
        "train_freq": (1, "episode"),
        "gradient_steps": 2000,   
        "tensorboard_log": LOG_DIR
    }

    ckpt = latest_checkpoint()
    
    if ckpt:
        print("Loading from latest checkpoint:", ckpt)
        model = SAC.load(
            ckpt,
            env=env,
            device="cuda",
            custom_objects=params 
        )
        
        replay_path = ckpt.replace(".zip", "_replay.pkl")
        migrate_buffer(model, replay_path)
        
        start_steps = int(ckpt.split("_")[-1].split(".")[0])

    else:
        print("Starting from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            **params
        )
        start_steps = 0

    total_steps = start_steps

    try:
        while total_steps < TOTAL_TIMESTEPS:
            model.learn(total_timesteps=STEPS_PER_SAVE, reset_num_timesteps=False)
            total_steps += STEPS_PER_SAVE

            path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_{total_steps}.zip")
            model.save(path)
            replay_path = path.replace(".zip", "_replay.pkl")
            model.save_replay_buffer(replay_path)

            print(f"Saved: {path} | Buffer: {model.replay_buffer.size()}/{model.replay_buffer.buffer_size}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    train()