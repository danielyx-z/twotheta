from stable_baselines3 import SAC
from esp_env import CartPoleESP32Env
import os

PORT = "COM7"
BAUD = 921600
MODEL_NAME = "single_pendulum"

def train():
    env = CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=500)

    # Check for existing model
    if os.path.exists(f"{MODEL_NAME}.zip"):
        print("Loading existing SAC model...")
        model = SAC.load(MODEL_NAME, env=env)
    else:
        print("Starting SAC training from scratch...")
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=5e-4,
            buffer_size=10000,
            learning_starts=100,
            batch_size=64,
            tau=0.005,
            gamma=0.99,
            ent_coef='auto',
            train_freq=(1, "episode"), # Wait for terminated=True or truncated=True
            gradient_steps=-1,         # Run 1 gradient step for every environment step collected
        )

    try:
        # Train for short bursts so we can save often
        steps_per_save = 1000
        total_steps = 0
        
        while total_steps < 50000:
            model.learn(total_timesteps=steps_per_save, reset_num_timesteps=False)
            total_steps += steps_per_save
            model.save(MODEL_NAME)
            print(f"--- Saved at {total_steps} steps ---")
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        model.save(MODEL_NAME)
        env.close()

if __name__ == "__main__":
    train()