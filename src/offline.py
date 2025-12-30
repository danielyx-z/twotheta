import os
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env # Needed for environment metadata

CKPT_DIR = "./checkpoints"
MODEL_NAME = "single_pendulum"
OFFLINE_STEPS = 50000  # Number of gradient updates to perform
BATCH_SIZE = 256

def latest_checkpoint():
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip")]
    if not files: return None
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CKPT_DIR, files[-1])

def train_offline():
    ckpt_path = latest_checkpoint()
    replay_path = ckpt_path.replace(".zip", "_replay.pkl")

    if not ckpt_path or not os.path.exists(replay_path):
        print("No checkpoint or replay buffer found!")
        return

    # 1. Load the model
    # We use a dummy env just to satisfy the SB3 loader requirements
    # max_steps=1 is fine as we won't actually call env.step()
    model = SAC.load(ckpt_path, device="cuda")
    
    # 2. Load the Replay Buffer
    print(f"Loading Replay Buffer from: {replay_path}")
    model.load_replay_buffer(replay_path)
    
    buffer = model.replay_buffer
    current_size = buffer.size()
    
    if current_size < BATCH_SIZE:
        print("Buffer too small for training!")
        return

    print(f"Starting Offline Training for {OFFLINE_STEPS} steps...")
    print(f"Buffer density: {current_size}/{buffer.buffer_size}")

    for i in range(OFFLINE_STEPS):
        model.train(batch_size=BATCH_SIZE, gradient_steps=1)
        
        if (i + 1) % 1000 == 0:
            print(f"Progress: {i + 1}/{OFFLINE_STEPS} updates completed.")

    new_path = ckpt_path.replace(".zip", "_offline_refined.zip")
    model.save(new_path)
    print(f"Offline training complete. Refined model saved to: {new_path}")

if __name__ == "__main__":
    train_offline()