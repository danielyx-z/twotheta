from stable_baselines3 import PPO
from esp_env import CartPoleESP32Env

PORT = "COM7"
BAUD = 921600

def train():
    env = CartPoleESP32Env(port=PORT, baudrate=BAUD)

    # Sensible architecture for continuous control
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=3e-4,
        n_steps=4096,      # Larger buffer for continuous stream
        batch_size=128,
        ent_coef=0.01,     # Encourage exploration since there are no resets
        policy_kwargs=policy_kwargs,
    )

    try:
        print("Training live... Press Ctrl+C to stop and save.")
        model.learn(total_timesteps=500000)
    except KeyboardInterrupt:
        print("\nSaving model...")
    
    model.save("ppo_cartpole_continuous")
    env.close()

if __name__ == "__main__":
    train()