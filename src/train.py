import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env

PORT = "COM8"
BAUD = 921600
MODEL_NAME = "single_pendulum"
LOG_DIR = "./tensorboard_logs/"

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=1000))

def train():
    env = DummyVecEnv([make_env])

    # Define the 64x64 architecture
    policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[64, 64]))

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print("Loading existing SAC model...")
        model = SAC.load(MODEL_NAME, env=env, device="cuda")
    else:
        print("Starting SAC training from scratch...")
        model = SAC(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            device="cuda",
            verbose=1,
            learning_rate=5e-4,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto_0.1',
            train_freq=(1, "episode"),
            gradient_steps=-1,
            tensorboard_log=LOG_DIR
        )

    try:
        steps_per_save = 1000
        total_steps = 0
        while total_steps < 50000:
            model.learn(total_timesteps=steps_per_save, reset_num_timesteps=False)
            total_steps += steps_per_save
            model.save(MODEL_NAME)
            print(f"--- Saved at {total_steps} total steps ---")

    except KeyboardInterrupt:
        print("\nStopping training...")
    finally:
        model.save(MODEL_NAME)
        env.close()

if __name__ == "__main__":
    train()