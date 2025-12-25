import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from esp_env import CartPoleESP32Env

# Configuration
PORT = "COM8"
BAUD = 921600
MODEL_NAME = "single_pendulum"
LOG_DIR = "./logs"

class MathPhaseTimerCallback(BaseCallback):
    """
    Callback to time only the gradient update (math) phase of SAC.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.start_math = 0

    def _on_training_start(self) -> None:
        # Triggered right before the gradient updates begin
        self.start_math = time.time()

    def _on_training_end(self) -> None:
        # Triggered right after the gradient updates finish
        duration = time.time() - self.start_math
        print(f"\n>>> MATH PHASE (Training) took: {duration:.4f} seconds")

    def _on_step(self) -> bool:
        return True

def make_env():
    # max_steps set to 1000 as per environment definition
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=1000))

def train():
    env = DummyVecEnv([make_env])

    if os.path.exists(f"{MODEL_NAME}.zip"):
        print("Loading existing SAC model...")
        model = SAC.load(MODEL_NAME, env=env, device="cuda")
    else:
        print("Starting SAC training from scratch...")
        model = SAC(
            "MlpPolicy",
            env,
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
        )

    # Initialize the timing callback
    timer_callback = MathPhaseTimerCallback()

    try:
        steps_per_save = 1000
        total_steps = 0

        while total_steps < 50000:
            # model.learn handles both data collection (Physical Phase) 
            # and gradient updates (Math Phase)
            model.learn(
                total_timesteps=steps_per_save, 
                reset_num_timesteps=False, 
                callback=timer_callback
            )
            
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