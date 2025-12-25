import gymnasium as gym
import numpy as np
from gymnasium import spaces
from esp_controller import ESP32SerialController
import math
import time

class CartPoleESP32Env(gym.Env):
    def __init__(self, port="COM7", baudrate=921600, max_steps=1000):
        super().__init__()
        self.esp = ESP32SerialController(port, baudrate)
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        self.max_pos = 7900.0  
        self.max_episode_steps = max_steps
        self.current_step = 0

    def _get_obs(self, state):
        t1, t2, v1, v2, pos = state
        return np.array([
            math.sin(t1), math.cos(t1),
            math.sin(t2), math.cos(t2),
            v1, v2, 
            pos / self.max_pos
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.esp.move(10.0) 
        
        start_wait = time.time()
        while True:
            if time.time() - start_wait > 5.0:
                print("Warning: Reset timed out!")
                self.esp.move(0.0)
                break

            state = self.esp.receive_state()

            if state is None:
                continue
                
            t1, t2, v1, v2, pos = state
            
            if abs(pos) < 50:
                self.esp.move(0.0)
                time.sleep(0.1) 
                break
            
            time.sleep(0.01)

        for _ in range(10):
            self.esp.receive_state()
            
        state = self.esp.receive_state()
        while state is None:
            state = self.esp.receive_state()

        return self._get_obs(state), {}

    def step(self, action):
        self.current_step += 1
        act = np.clip(action[0], -1.0, 1.0)

        self.esp.move(float(act))
        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()

        obs = self._get_obs(raw_state)
        t1, t2, v1, v2, pos = raw_state
        
        upright_reward = -math.cos(t1) + 0.5
        dist_penalty = (pos / self.max_pos) ** 2
        
        reward = float(upright_reward - (0.01 * dist_penalty))
        
        terminated = False
        if abs(pos) > self.max_pos:
            terminated = True
            print("Episode terminated.")
            reward -= 10.0 
            self.esp.move(0.0)

        truncated = False
        if self.current_step >= self.max_episode_steps:
            truncated = True
            print("Episode length truncated.")
            self.esp.move(0.0)

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()