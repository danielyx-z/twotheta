import os
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from esp_env import CartPoleESP32Env # Assuming this is your local file

# --- CONFIGURATION ---
PORT = "COM8"
BAUD = 921600
CKPT_DIR = "./checkpoints"
REPLAY_BUFFER_PATH = os.path.join(CKPT_DIR, "single_pendulum_198000_replay.pkl")
SAVE_NAME_MODEL = os.path.join(CKPT_DIR, "single_pendulum_0.zip")
SAVE_NAME_REPLAY = os.path.join(CKPT_DIR, "single_pendulum_0_replay.pkl")

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    # max_steps=1000 to match your previous setup
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=1000))

def main():
    # 1. Create the Environment
    env = DummyVecEnv([make_env])

    # 2. Define fresh network architecture
    policy_kwargs = dict(net_arch=dict(pi=[128, 64], qf=[128, 64]))

    params = {
        "learning_rate": 3e-4,
        "buffer_size": 80000,
        "learning_starts": 1000, # Start learning almost immediately since buffer is full
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "ent_coef": "auto_0.1",
        "train_freq": (1, "episode"),
        "gradient_steps": 1500,
    }

    # 3. Initialize FRESH model (New Actor/Critic)
    print("Creating fresh Actor and Critic networks...")
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        device="cuda",
        verbose=1,
        **params
    )

    # 4. Load the Replay Buffer
    if os.path.exists(REPLAY_BUFFER_PATH):
        print(f"Loading replay buffer from: {REPLAY_BUFFER_PATH}")
        model.load_replay_buffer(REPLAY_BUFFER_PATH)
        print(f"Buffer successfully loaded! Size: {model.replay_buffer.size()}")
    else:
        print(f"ERROR: Replay buffer not found at {REPLAY_BUFFER_PATH}")
        return

    # 5. Save everything at 'Step 0'
    print(f"Saving step 0 model to: {SAVE_NAME_MODEL}")
    model.save(SAVE_NAME_MODEL)
    
    print(f"Saving step 0 replay buffer to: {SAVE_NAME_REPLAY}")
    model.save_replay_buffer(SAVE_NAME_REPLAY)

    print("\nInitialization Complete. You now have a fresh brain with old memories.")
    env.close()

if __name__ == "__main__":
    main()