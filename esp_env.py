import gymnasium as gym
import numpy as np
from gymnasium import spaces
from esp_controller import ESP32SerialController
import math
import time

class CartPoleESP32Env(gym.Env):
    def __init__(self, port="/dev/ttyUSB0", baudrate=921600, max_steps=1000):
        super().__init__()
        self.esp = ESP32SerialController(port, baudrate)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # [sin(t1), cos(t1), error, v1, pos, motor_vel, dt, prev_action]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.max_pos = 31900.0  
        self.max_motor_speed = 90000.0
        self.max_episode_steps = max_steps
        self.current_step = 0
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 20
        self.SPIN_THRESHOLD = 3 * math.pi 
        self.last_step_time = time.perf_counter()
        self.first_step_of_episode = True
        
        # Track previous action for observation and reward calculation
        self.prev_action = 0.0

    def _get_obs(self, state, dt_measured, prev_action):
        t1, t2, v1, v2, pos, motor_vel = state
        return np.array([
            math.sin(t1), 
            math.cos(t1),
            ((t1 + math.pi) % (2 * math.pi) - math.pi) / math.pi,
            v1 / self.SPIN_THRESHOLD,
            pos / self.max_pos,
            motor_vel / self.max_motor_speed,
            dt_measured * 60,
            float(prev_action) # Add previous action to observation
        ], dtype=np.float32)

    def _calculate_reward(self, state, action, prev_action, terminated):
        t1, v1, pos = state[0], state[2], state[4]
        
        if terminated:
            return -20.0 

        # 1. Uprightness Reward
        error = (t1 + math.pi) % (2 * math.pi) - math.pi
        r_base = (math.cos(error) + 1) / 2
        
        sigma_angle = 0.1 
        sigma_vel = 0.05
        r_stability = math.exp(-(error**2) / (2 * sigma_angle**2)) * math.exp(-(v1**2) / (2 * sigma_vel**2))

        # 2. Velocity Penalty (penalize high velocity when upright)
        uprightness = r_base ** 2
        r_velocity = -0.05 * uprightness * (v1 ** 2)

        # 3. Energy/Magnitude Penalty
        current_action = float(np.asarray(action).item())
        r_action = -0.01 * abs(current_action)

        # 4. Position Penalty (Keep near center)
        r_pos = -0.2 * (abs(pos) / self.max_pos) ** 2
        
        # 5. Delta Action Penalty (New: Smoothness)
        # Penalize large changes in action to prevent high frequency oscillation
        delta_action = current_action - prev_action
        r_delta = -0.5 * (delta_action ** 2)

        return float(np.asarray(r_base + r_stability + r_velocity + r_action + r_pos + r_delta).item())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.esp.serial.reset_output_buffer()
        self.active_damp()
        self.esp.move(10.0) 
        self.esp.serial.reset_input_buffer()

        self.current_step = 0
        self.first_step_of_episode = True
        self.prev_action = 0.0 # Reset previous action
        
        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()
            
        # Initial observation has prev_action = 0.0
        obs = self._get_obs(raw_state, 0.018, 0.0)
        self.last_step_time = time.perf_counter()

        time.sleep(0.5)
        return obs, {}

    def active_damp(self):
        start_time = time.time()
        last_move = time.time()
        stabilized = 0
        while True:
            if time.time() - start_time > 60.0:
                print("Reset Timeout! Forcing start...")
                break

            state = self.esp.receive_state()
            while state is None:
                time.sleep(0.001)
                state = self.esp.receive_state()

            t1, v1, pos = state[0], state[2], state[4]
            u_energy = 0.2 * v1 * math.cos(t1)
            u_center = 0.01 * (pos / self.max_pos)
            action = np.clip(u_energy + u_center, -0.8, 0.8)

            if abs(v1) < 0.2 and math.cos(t1) < 0.97:
                stabilized += 1
                if stabilized > 30:  
                    break
            else:
                stabilized = 0

            if time.time() - last_move > 0.1:
                self.esp.move(float(action))
                last_move = time.time()
                
            time.sleep(0.01)   
        self.esp.move(0.0) 

    def step(self, action):
        raw_action = np.clip(action[0], -1.0, 1.0)
        current_action = np.sign(raw_action) * (abs(raw_action)**1.5)
        self.current_step += 1

        if self.first_step_of_episode:
            self.esp.serial.reset_input_buffer()
            raw_state = self.esp.receive_state()
            while raw_state is None:
                raw_state = self.esp.receive_state()
            self.first_step_of_episode = False

        self.last_step_time = time.perf_counter()
        self.esp.move(float(current_action))

        self.esp.serial.reset_input_buffer()
        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()

        # Termination logic
        angular_vel = abs(raw_state[2]) 
        if angular_vel > self.SPIN_THRESHOLD:
            self.overspeed_counter += 1
        else:
            self.overspeed_counter = 0 

        hit_wall = abs(raw_state[4]) > self.max_pos
        is_spinning = self.overspeed_counter > self.MAX_OVERSPEED_FRAMES
        
        terminated = hit_wall or is_spinning
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated and is_spinning:
            print("Terminating episode due to overspeed.")

        # Calculate reward
        reward = self._calculate_reward(raw_state, current_action, self.prev_action, terminated)
        
        current_time = time.perf_counter()
        actual_dt = current_time - self.last_step_time
        
        #print(actual_dt)
        # Update prev_action for the *next* step's calculation
        # The observation returned here is for S_{t+1}, so the "previous action" 
        # relative to S_{t+1} is the current_action we just took.
        obs = self._get_obs(raw_state, actual_dt, current_action)
        self.prev_action = current_action
        
        self.last_step_time = current_time

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()

if __name__ == "__main__":
    # Test execution
    test = CartPoleESP32Env()
    print("swing it")
    time.sleep(2)
    test.active_damp()
    print("dampened")
    test.close()