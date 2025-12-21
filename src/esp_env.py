import gymnasium as gym
import numpy as np
from gymnasium import spaces
from esp_controller import ESP32SerialController
import math
import time

class CartPoleESP32Env(gym.Env):
    def __init__(self, port="COM7", baudrate=921600):
        super().__init__()
        self.esp = ESP32SerialController(port, baudrate)
        
        # Action: Speed (-1.0 to 1.0)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Obs: sin/cos(t1), sin/cos(t2), v1, v2, pos_norm, target_t1, target_t2 (9 total)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)
        
        self.max_pos = 2000.0  
        self.target_t1 = 3.1415   # Vertical Upright
        self.target_t2 = 0.0   

    def _get_obs(self, state):
        t2, t1, v1, v2, pos = state
        return np.array([
            math.sin(t1), math.cos(t1),
            math.sin(t2), math.cos(t2),
            v1, v2,
            pos / self.max_pos,
            self.target_t1,
            self.target_t2
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.esp.receive_state()
        while state is None:
            state = self.esp.receive_state()
            time.sleep(0.001)
        return self._get_obs(state), {}

    def step(self, action):
        start_time = time.perf_counter()
        
        self.esp.move(float(action[0]))
        
        # Busy-wait for 1ms loop timing
        while (time.perf_counter() - start_time) < 0.001:
            pass
            
        raw_state = self.esp.receive_state()
        if raw_state is None:
            return np.zeros(9, dtype=np.float32), 0.0, False, False, {}

        obs = self._get_obs(raw_state)
        t2, t1, v1, v2, pos = raw_state

        # Reward Logic
        # 1. Distance from target angles (using cosine similarity/difference)
        # For upright, target_t1=0, so cos(t1 - 0) works well.
        reward = math.cos(t1 - self.target_t1) 
        
        # 2. Centering (only if you want it to stay near middle)
        reward -= 0.1 * (pos / self.max_pos)**2 
        
        # 3. Action Penalty (prevents jitter/high-frequency oscillation)
        reward -= 0.01 * (action[0]**2)

        # No termination: always return False for terminated/truncated
        return obs, reward, False, False, {}

    def close(self):
        self.esp.close()