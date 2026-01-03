import os
import time
import numpy as np
import torch
from sbx import TQC
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "/dev/ttyUSB0"
BAUD = 921600
CKPT_DIR = "./checkpoints"
MODEL_NAME = "tqc_pendulum_sbx"
MAX_EVAL_STEPS = 3000

def get_latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None
    # Filter for the .zip model files
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip") and MODEL_NAME in f]
    if not files:
        return None
    
    # Sort by step count (assuming naming convention model_name_XXXX_steps.zip)
    def extract_steps(filename):
        try:
            return int("".join(filter(str.isdigit, filename)))
        except ValueError:
            return 0

    files.sort(key=extract_steps)
    return os.path.join(CKPT_DIR, files[-1])

def run_evaluation():
    env = CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=MAX_EVAL_STEPS)

    ckpt_path = get_latest_checkpoint()
    if ckpt_path is None:
        print("No checkpoint found in ./checkpoints. Exiting.")
        return

    print(f"Loading model: {ckpt_path}")
    model = TQC.load(ckpt_path, device="cpu")

    try:    
        print("\n--- Starting New Evaluation Episode ---")
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        step_count = 0
        

        start_time = time.perf_counter()

        while not (done or truncated):
            # Predict DETERMINISTIC action (No entropy/noise)
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step_count % 100 == 0:
                fps = step_count / (time.perf_counter() - start_time)
                print(f"Step: {step_count} | FPS: {fps:.1f} | Reward: {total_reward:.1f}")

        print(f"Episode Finished. Steps: {step_count} | Total Reward: {total_reward:.2f}")
        

    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    run_evaluation()