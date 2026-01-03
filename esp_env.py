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

        # Action: -1.0 to 1.0 (Frequency)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # [sin(t), cos(t), error_from_top, v, pos, motor_v, dt, prev_act]
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        
        self.max_pos = 31900.0  
        self.max_motor_speed = 75000.0
        self.max_episode_steps = max_steps
        self.current_step = 0
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 20
        self.SPIN_THRESHOLD = 3 * math.pi 
        self.last_step_time = time.perf_counter()
        self.first_step_of_episode = True
        
        self.prev_action = 0.0

    def _get_obs(self, state, dt_measured, prev_action):
        t1, t2, v1, v2, pos, motor_vel = state
        
        # We assume C++ has calibrated so Bottom is ~0 and Top is ~PI
        # Error calculation: Distance from PI
        error_from_top = math.atan2(math.sin(t1 - math.pi), math.cos(t1 - math.pi))
        return np.array([
            math.sin(t1),       
            math.cos(t1),      
            error_from_top / math.pi, # Normalized error
            v1 / self.SPIN_THRESHOLD,
            pos / self.max_pos,
            motor_vel / self.max_motor_speed,
            dt_measured * 60,
            float(prev_action)
        ], dtype=np.float32)

    def _calculate_reward(self, state, action, prev_action, terminated):
        t1, v1, pos = state[0], state[2], state[4]
        
        if terminated:
            return -20.0 

        # Error relative to PI (180 deg)
        # Using (t1 - pi) logic assuming t1 comes in around 3.14 at top
        error = math.atan2(math.sin(t1 - math.pi), math.cos(t1 - math.pi))
        
        # 1. Base Angle Reward (Wide tolerance)
        # We accept a wider range as "good" so it doesn't fight the physical balance point
        r_base = (1 - math.cos(t1) ) / 2
        
        # 2. Stability Reward (The "King" Reward)
        # We reward High Uprightness AND Low Velocity heavily.
        # If angle is 178 deg but velocity is 0, this will still be very high.
        sigma_angle = 0.15 # Widened slightly to accept 178-182 deg
        sigma_vel = 0.05   # Strict velocity requirement
        
        r_stability = math.exp(-(error**2) / (2 * sigma_angle**2)) * \
                      math.exp(-(v1**2) / (2 * sigma_vel**2))

        # 3. Penalties
        uprightness = r_base ** 2
        r_velocity = -0.05 * uprightness * (v1 ** 2)
        
        current_action = float(np.asarray(action).item())
        r_action = -0.01 * abs(current_action)
        r_pos = -0.1 * (abs(pos) / self.max_pos) ** 2
        
        delta_action = current_action - prev_action
        r_delta = -0.5 * (delta_action ** 2)

        return float(np.asarray(r_base + r_stability + r_velocity + r_action + r_pos + r_delta).item())
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.esp.serial.reset_output_buffer()
        self.active_damp() # Your guaranteed logic
        self.esp.move(10.0) 

        self.esp.serial.reset_input_buffer()
        self.current_step = 0
        self.first_step_of_episode = True
        self.prev_action = 0.0 
        
        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()
            
        # Standard observation (Assuming C++ handled offset)
        obs = self._get_obs(raw_state, 0.018, 0.0)
        
        #print(f"Start Angle: {math.degrees(raw_state[0]):.2f}")

        self.last_step_time = time.perf_counter()
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
            action = -np.clip(u_energy + u_center, -0.8, 0.8)

            if abs(v1) < 0.2 and math.cos(t1) > 0.97:
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

        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()


        hit_wall = abs(raw_state[4]) > self.max_pos
        if abs(raw_state[2]) > self.SPIN_THRESHOLD:
            self.overspeed_counter += 1
        else:
            self.overspeed_counter = 0

        is_spinning = self.overspeed_counter > self.MAX_OVERSPEED_FRAMES
        
        terminated = hit_wall or is_spinning
        truncated = self.current_step >= self.max_episode_steps
        
        if terminated and is_spinning:
            print("Terminating due to spin.")

        reward = self._calculate_reward(raw_state, raw_action, self.prev_action, terminated)
        
        current_time = time.perf_counter()
        actual_dt = current_time - self.last_step_time
        #print(actual_dt)
        obs = self._get_obs(raw_state, actual_dt, raw_action)
        self.prev_action = raw_action # Note: storing RAW action for history
        self.last_step_time = current_time

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()

if __name__ == "__main__":
    test = CartPoleESP32Env()
    test.active_damp()
    print("dampened")
    test.close()