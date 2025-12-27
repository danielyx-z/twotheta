import os
from sbx import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

PORT = "COM8"
BAUD = 921600
MODEL_NAME = "sbx_pendulum"
LOG_DIR = "./sbx_logs/"
CKPT_DIR = "./sbx_checkpoints"
TOTAL_TIMESTEPS = 200000
STEPS_PER_SAVE = 500

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=600))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    valid_files = [f for f in files if f.replace(".zip", "").split("_")[-1].isdigit()]
    if not valid_files:
        return None
    valid_files.sort(key=lambda x: int(x.replace(".zip", "").split("_")[-1]))
    return os.path.join(CKPT_DIR, valid_files[-1])

def train():
    env = DummyVecEnv([make_env])
    
    params = {
        "learning_rate": 5e-4,
        "buffer_size": 20000,
        "batch_size": 128,
        "learning_starts": 1000,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto",
        "train_freq": (1, "episode"),
        "gradient_steps": 1500,
        "tensorboard_log": LOG_DIR
    }

    ckpt = latest_checkpoint()
    
    if ckpt:
        print(f"Loading SBX model: {ckpt}")
        model = SAC.load(
            ckpt, 
            env=env, 
            custom_objects=params
        )
        try:
            start_steps = int(ckpt.replace(".zip", "").split("_")[-1])
        except:
            start_steps = 0
    else:
        print("Starting fresh SBX training")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            **params
        )
        start_steps = 0

    total_steps = start_steps

    try:
        while total_steps < TOTAL_TIMESTEPS:
            model.learn(total_timesteps=STEPS_PER_SAVE, reset_num_timesteps=False)
            total_steps += STEPS_PER_SAVE
            path = os.path.join(CKPT_DIR, f"{MODEL_NAME}_{total_steps}")
            model.save(path)
            print(f"Saved: {path}")
    except KeyboardInterrupt:
        model.save(os.path.join(CKPT_DIR, f"{MODEL_NAME}_interrupted"))
    finally:
        env.close()

if __name__ == "__main__":
    train()