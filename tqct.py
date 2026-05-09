import os
import time
from typing import Callable
from sbx import TQC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "/dev/ttyUSB0"
BAUD = 921600
MODEL_NAME = "tqc_pendulum_sbx"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints"
TOTAL_TIMESTEPS = 150000 
STEPS_PER_SAVE = 10000

os.makedirs(CKPT_DIR, exist_ok=True)

def linear_schedule(initial_value: float, final_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule that clips at final_value.
    """
    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 to 0.0 over TOTAL_TIMESTEPS.
        # max(0, ...) ensures LR stays at final_value if steps exceed TOTAL_TIMESTEPS.
        progress = max(0.0, progress_remaining)
        return final_value + (initial_value - final_value) * progress
    return func

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=3000))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None, 0
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip") and MODEL_NAME in f]
    if not files:
        return None, 0
    valid_files = []
    for f in files:
        try:
            parts = f.replace(".zip", "").split("_")
            if "steps" in parts:
                idx = parts.index("steps")
                num = int(parts[idx-1])
            else:
                num = int("".join(filter(str.isdigit, f)))
            valid_files.append((num, f))
        except (ValueError, IndexError):
            continue
    if not valid_files:
        return None, 0
    valid_files.sort(key=lambda x: x[0])
    best_num, best_file = valid_files[-1]
    return os.path.join(CKPT_DIR, best_file), best_num

def train():
    env = DummyVecEnv([make_env])

    # Schedule from 0.0002 to 0.00005
    lr_schedule = linear_schedule(0.0002, 0.00005)

    policy_kwargs = dict(
        net_arch=[256, 256],
        n_quantiles=25,
    )

    params = {
        "learning_rate": 0.00005,
        "buffer_size": 100000, 
        "learning_starts": 2000, 
        "batch_size": 512, 
        "tau": 0.001,
        "gamma": 0.99,
        "ent_coef": "auto",
        "train_freq": (1, "step"),
        "gradient_steps": 20,
        "top_quantiles_to_drop_per_net": 2,
        "tensorboard_log": LOG_DIR
    }

    checkpoint_callback = CheckpointCallback(
        save_freq=STEPS_PER_SAVE,
        save_path=CKPT_DIR,
        name_prefix=MODEL_NAME,
        save_replay_buffer=True
    )

    ckpt_path, start_steps = latest_checkpoint()
    
    if ckpt_path:
        print(f"--- LOADING TQC CHECKPOINT: {ckpt_path} ---")
        model = TQC.load(ckpt_path, env=env, tensorboard_log=LOG_DIR, custom_objects=params)
        replay_name = f"{MODEL_NAME}_replay_buffer_{start_steps}_steps.pkl"
        replay_path = os.path.join(CKPT_DIR, replay_name)
        if os.path.exists(replay_path):
            print(f"Loaded replay buffer: {replay_path}")
            model.load_replay_buffer(replay_path)

    else:
        print("--- Starting TQC from scratch ---")
        model = TQC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params
        )
        start_steps = 0

    try:
        print(f"Begin training.")
        run_name = f"TQC_plswork"
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            reset_num_timesteps=False,
            tb_log_name=run_name
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train()