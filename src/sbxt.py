import os
from sbx import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

# --- Configuration ---
PORT = "/dev/ttyUSB0"
BAUD = 921600
MODEL_NAME = "single_pendulum_sbx"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints"
TOTAL_TIMESTEPS = 500000
STEPS_PER_SAVE = 3000

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    # Ensure you have permissions for /dev/ttyUSB0 (sudo chmod 666 /dev/ttyUSB0)
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=3000, enable_viz=False))

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

    # SBX uses a slightly different architecture definition
    policy_kwargs = dict(
        net_arch=[128, 128] # SBX defaults to ReLU; pi and qf share this arch usually
    )

    params = {
        "learning_rate": 3e-4,
        "buffer_size": 80000, 
        "learning_starts": 5000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto", # SBX 'auto' works slightly differently but usually better
        "train_freq": (1, "step"),
        "gradient_steps": 5,   
        "tensorboard_log": LOG_DIR
    }

    ckpt = latest_checkpoint()
    
    if ckpt:
        print(f"Loading SBX checkpoint: {ckpt}")
        model = SAC.load(ckpt, env=env)
        replay_path = ckpt.replace(".zip", "_replay.pkl")
        if os.path.exists(replay_path):
            model.load_replay_buffer(replay_path)
            print("Replay buffer loaded.")
        
        start_steps = int(ckpt.split("_")[-1].split(".")[0])
    else:
        print("Starting SBX from scratch")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params
        )
        start_steps = 0

    total_steps = start_steps

    try:
        while total_steps < TOTAL_TIMESTEPS:
            # model.learn in SBX is JIT compiled, so the first call might be slow
            model.learn(total_timesteps=STEPS_PER_SAVE, reset_num_timesteps=False)
            total_steps += STEPS_PER_SAVE

            path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_{total_steps}.zip")
            model.save(path)
            model.save_replay_buffer(path.replace(".zip", "_replay.pkl"))

            print(f"Saved: {path}")

    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train()