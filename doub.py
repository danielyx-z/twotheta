import os
import jax
import jax.numpy as jnp
from flax import linen as nn
from sbx import TQC
from sbx.tqc.policies import TQCPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from esp_double_env import CartPoleESP32Env
class CustomLayerNormPolicy(TQCPolicy):
    def build_q_net(self): # Type hint removed to prevent ImportError
        class QNet(nn.Module):
            net_arch: list[int]
            n_quantiles: int

            @nn.compact
            def __call__(self, obs: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
                x = jnp.concatenate([obs, action], axis=-1)
                for n_units in self.net_arch:
                    x = nn.Dense(n_units)(x)
                    x = nn.LayerNorm()(x) 
                    x = nn.relu(x)
                return nn.Dense(self.n_quantiles)(x)
        
        return QNet(net_arch=self.net_arch, n_quantiles=self.n_quantiles)

    def build_actor(self): # Type hint removed
        class Actor(nn.Module):
            net_arch: list[int]
            action_dim: int

            @nn.compact
            def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
                x = obs
                for n_units in self.net_arch:
                    x = nn.Dense(n_units)(x)
                    x = nn.LayerNorm()(x) 
                    x = nn.relu(x)
                
                # SBX expects mean and log_std for the Gaussian policy
                mean = nn.Dense(self.action_dim)(x)
                log_std = nn.Dense(self.action_dim)(x)
                return mean, log_std
        
        return Actor(net_arch=self.net_arch, action_dim=self.action_dim)
# --- 2. CONFIGURATION ---
PORT = "/dev/ttyUSB0"
BAUD = 921600
MODEL_NAME = "tqc_dpendulum_sbx_ln"
LOG_DIR = "./tensorboard_logs/"
CKPT_DIR = "./double_checkpoints"
TOTAL_TIMESTEPS = 150000
STEPS_PER_SAVE = 6000

os.makedirs(CKPT_DIR, exist_ok=True)

def make_env():
    return Monitor(CartPoleESP32Env(port=PORT, baudrate=BAUD, max_steps=3000))

def latest_checkpoint():
    if not os.path.exists(CKPT_DIR):
        return None, 0
    files = [f for f in os.listdir(CKPT_DIR) if f.endswith(".zip") and MODEL_NAME in f]
    if not files:
        return None, 0
    valid_files = [(int(f.split('_')[-2]), f) for f in files if 'steps' in f]
    if not valid_files: return None, 0
    valid_files.sort(key=lambda x: x[0])
    return os.path.join(CKPT_DIR, valid_files[-1][1]), valid_files[-1][0]

def train():
    env = DummyVecEnv([make_env])
    
    policy_kwargs = dict(
        net_arch=[256, 256],
        n_quantiles=25
    )

    # Aggressive 10 UTD Params
    params = {
        "learning_rate": 3e-4, 
        "buffer_size": 150000, 
        "learning_starts": 2000, 
        "batch_size": 256,      # Lower batch size often helps high UTD stability
        "tau": 0.005,
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
        print(f"--- LOADING TQC + LN CHECKPOINT: {ckpt_path} ---")
        model = TQC.load(ckpt_path, env=env, tensorboard_log=LOG_DIR, custom_objects=params)
        model.load_replay_buffer(os.path.join(CKPT_DIR, f"{MODEL_NAME}_replay_buffer_{start_steps}_steps.pkl"))
    else:
        print("--- Starting TQC with LayerNorm from scratch ---")
        model = TQC(
            CustomLayerNormPolicy,  # <--- Pass the custom class here
            env,
            policy_kwargs=policy_kwargs,
            verbose=1,
            **params
        )

    try:
        model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, reset_num_timesteps=False)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()

if __name__ == "__main__":
    train()