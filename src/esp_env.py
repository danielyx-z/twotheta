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
        self.is_initialized = False
        self.dt = 0.015 #15ms
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 30
        self.SPIN_THRESHOLD = 3 * math.pi #rad / s
        self.last_step_time = time.perf_counter()

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

        upright_reward = (-math.cos(t1 - 0.0275) + 1) ** 2
        dist_penalty = 0.5 * (pos / self.max_pos) ** 2
        velocity_penalty = 0.002 * (v1**2)
        action_penalty = 0.01 * (action**2)

        reward = upright_reward - dist_penalty - velocity_penalty - action_penalty
        return float(reward)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        if not self.is_initialized:
            self.esp.move(10.0) # Trigger hardware homing
            self.is_initialized = True

        #will already be actively dampened and homed. can settle in training time.
        self.esp.serial.reset_input_buffer()
        state = self.esp.receive_state()
        while state is None: 
            state = self.esp.receive_state()

        return self._get_obs(state), {}

    def active_damp(self):
        start_time = time.time()
        last_move = time.time()
        stabilized = 0
        while True:
            if time.time() - start_time > 90.0:
                print("Reset Timeout! Forcing start...")
                break

            state = self.esp.receive_state()
            if state is None: 
                time.sleep(0.001)
                continue

            t1, v1, pos = state[0], state[2], state[4]
            u_energy = 0.2 * v1 * math.cos(t1)
            u_center = 0.1 * (pos / self.max_pos)
            action = -np.clip(u_energy + u_center, -0.9, 0.9)

            if abs(v1) < 0.3 and math.cos(t1) > 0.97:
                stabilized += 1
                action = u_center
                if stabilized > 50:  #stable for enough time
                    break
            if time.time() - last_move > 0.2:
                self.esp.move(float(action))
                last_move = time.time()

            time.sleep(0.01)      
        self.esp.move(0.0) 

    def step(self, action):
        while time.perf_counter() - self.last_step_time < self.dt:
            time.sleep(0.0001)
        
        current_time = time.perf_counter()
        self.last_step_time = current_time

        self.current_step += 1
        act = np.clip(action[0], -1.0, 1.0)

        self.esp.move(float(act))

        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()


        angular_vel = abs(raw_state[2]) 
        if angular_vel > self.SPIN_THRESHOLD:
            self.overspeed_counter += 1
        else:
            self.overspeed_counter = 0 

        hit_wall = abs(raw_state[4]) > self.max_pos
        is_spinning = self.overspeed_counter > self.MAX_OVERSPEED_FRAMES
        
        terminated = hit_wall or is_spinning
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated or truncated:
            if is_spinning:
                print("Terminating episode due to overspeed.")
            self.active_damp()
            self.esp.move(10)
            
        reward = self._calculate_reward(raw_state, act, terminated)
        
        return self._get_obs(raw_state), reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()