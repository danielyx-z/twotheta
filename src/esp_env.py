import gymnasium as gym
import numpy as np
from gymnasium import spaces
from esp_controller import ESP32SerialController
import math
import time

class CartPoleESP32Env(gym.Env):
    def __init__(self, port="COM8", baudrate=921600, max_steps=1000):
        super().__init__()
        self.esp = ESP32SerialController(port, baudrate)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self.max_pos = 31900.0  
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

    def _calculate_reward(self, state, action, terminated):
        t1, t2, v1, v2, pos = state
        if terminated:
            return -20.0

        upright_reward = -math.cos(t1 - 0.0275) 
        dist_penalty = 0.5 * (pos / self.max_pos) ** 2
        velocity_penalty = 0.00 * (v1**2)
        action_penalty = 0.01 * (action**2)

        reward = upright_reward - dist_penalty - velocity_penalty - action_penalty
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.esp.move(10.0) 
        time.sleep(5) 
        self.esp.serial.reset_input_buffer()
        state = None
        while True:
            state = self.esp.receive_state()
            if state is not None:
                v1 = state[2]
                if abs(v1) < 0.01:
                    break
            time.sleep(0.5)

        self.esp.serial.reset_input_buffer()
        state = self.esp.receive_state()
        while state is None:
            state = self.esp.receive_state()

        return self._get_obs(state), {}

    def step(self, action):
        self.current_step += 1
        act = np.clip(action[0], -1.0, 1.0)

        self.esp.serial.reset_input_buffer()
        self.esp.move(float(act))

        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()

        terminated = abs(raw_state[4]) > self.max_pos
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated or truncated:
            self.esp.move(0.0)

        reward = self._calculate_reward(raw_state, act, terminated)
        return self._get_obs(raw_state), reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()