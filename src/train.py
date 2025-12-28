import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

PORT = "COM8"
BAUD = 921600
MODEL_NAME = "single_pendulum"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./checkpoints"
TOTAL_TIMESTEPS = 500000
STEPS_PER_SAVE = 3000

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=1000))

def latest_checkpoint():
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip")]
    if not files:
        return None
    files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return os.path.join(CKPT_DIR, files[-1])

def train():
    env = DummyVecEnv([make_env])

    policy_kwargs = dict(net_arch=dict(pi=[128, 64], qf=[128, 64]))

    params = {
        "learning_rate": 3e-4,
        "buffer_size": 80000,
        "learning_starts": 3000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto_0.1",
        "train_freq": (1, "episode"),
        "gradient_steps": 1200,
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

        start_steps = int(ckpt.split("_")[-1].split(".")[0])
        # Load replay buffer if exists
        replay_path = ckpt.replace(".zip", "_replay.pkl")
        if os.path.exists(replay_path):
            print("Loading replay buffer:", replay_path)
            model.load_replay_buffer(replay_path)
        buffer = model.replay_buffer
        print(f"Replay buffer size: {buffer.size()}/{buffer.buffer_size}")

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

            buffer = model.replay_buffer
            print(f"Saved checkpoint: {path} | Replay buffer size: {buffer.size()}/{buffer.buffer_size}")

    except KeyboardInterrupt:
        pass
    finally:
        env.close()

if __name__ == "__main__":
    train()
