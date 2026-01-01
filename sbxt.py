import os
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
STEPS_PER_SAVE = 6000

# Global gatekeeper to stop double-reset during JAX JIT
RESET_COUNT = 0

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=3000, enable_viz=False))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None, 0
    
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip") and MODEL_NAME in f]
    if not files:
        return None, 0
    
    valid_files = []
    for f in files:
        try:
            # Extract number from "MODEL_NAME_XXXX_steps.zip"
            # We split by "_" and look for the segment before "steps"
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
    global RESET_COUNT
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(
        net_arch=[128, 64],
        dropout_rate=0.01,
        layer_norm=True
    )

    params = {
        "learning_rate": 3e-4,
        "buffer_size": 100000, 
        "learning_starts": 1000, 
        "batch_size": 512,
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

    ckpt_path, start_steps = latest_checkpoint()
    
    if ckpt_path:
        print(f"--- LOADING CHECKPOINT: {ckpt_path} (Starting at {start_steps} steps) ---")
        model = SAC.load(ckpt_path, env=env, tensorboard_log=LOG_DIR, custom_objects=params)

        # Replay buffer naming: droq_pendulum_sbx_replay_buffer_XXXX_steps.pkl
        replay_name = f"{MODEL_NAME}_replay_buffer_{start_steps}_steps.pkl"
        replay_path = os.path.join(CKPT_DIR, replay_name)
        
        if os.path.exists(replay_path):
            print(f"Loaded replay buffer: {replay_path}")
            model.load_replay_buffer(replay_path)

    else:
        print("--- Starting DroQ from scratch ---")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params
        )
        start_steps = 0

    try:
        print(f"Begin training from step {start_steps}. Wait for JAX...")
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            reset_num_timesteps=False # Keep the step count moving forward
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train()