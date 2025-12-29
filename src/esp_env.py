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
        # Reduced to 5: sin(t1), cos(t1), v1, pos_norm, dt
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        self.max_pos = 31900.0  
        self.max_episode_steps = max_steps
        self.current_step = 0
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 20
        self.SPIN_THRESHOLD = 3 * math.pi #rad / s
        self.last_step_time = time.perf_counter()
        self.first_step_of_episode = True

    def _get_obs(self, state, dt_measured):
        t1, t2, v1, v2, pos = state
        return np.array([
            math.sin(t1), math.cos(t1),
            v1, 
            pos / self.max_pos,
            dt_measured * 60
        ], dtype=np.float32)

    def _calculate_reward(self, state, action, terminated):
        # t1: 0=Bottom, pi=Top | v1: velocity | pos: cart position
        t1, v1, pos = state[0], state[2], state[4]

        if terminated:
            return -30.0 

        error = (t1 - math.pi + math.pi) % (2 * math.pi) - math.pi
        r_swingup = -math.cos(t1) + 1
        
        # Smooth Gaussian for the top
        r_precision = 2.0 * math.exp(-(error ** 2) / (2 * 0.3 ** 2))

        r_pos = -0.01 * abs(pos / self.max_pos)

        uprightness = max(0, -math.cos(t1)) # Only positive when pole is above horizontal
        r_velocity = -0.08 * uprightness * (v1 ** 2)

        return float(r_swingup + r_precision + r_pos + r_velocity)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.esp.serial.reset_output_buffer()
        self.first_step_of_episode = True
        self.esp.move(10.0) # Trigger hardware homing

        self.last_step_time = time.perf_counter()

        # Dummy state
        return self._get_obs(np.zeros(5, dtype=np.float32), 0.015), {}

    def active_damp(self):
        start_time = time.time()
        last_move = time.time()
        stabilized = 0
        while True:
            if time.time() - start_time > 90.0:
                print("Reset Timeout! Forcing start...")
                break

            state = self.esp.receive_state()
            while state is None:
                state = self.esp.receive_state()

            t1, v1, pos = state[0], state[2], state[4]
            u_energy = 0.2 * v1 * math.cos(t1)
            u_center = 0.01 * (pos / self.max_pos)
            action = -np.clip(u_energy + u_center, -0.8, 0.8)

            if abs(v1) < 0.2 and math.cos(t1) > 0.98:
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
        if self.first_step_of_episode:
            self.esp.serial.reset_input_buffer()
            raw_state = self.esp.receive_state()
            while raw_state is None:
                raw_state = self.esp.receive_state()
            
            self.first_step_of_episode = False
            # Update timing so the first 'dt' isn't huge (e.g., 3 seconds)
            self.last_step_time = time.perf_counter()


        act = np.clip(action[0], -1.0, 1.0)
        self.esp.move(float(act))

        current_time = time.perf_counter()
        actual_dt = current_time - self.last_step_time
        self.last_step_time = current_time

        if 1 < actual_dt < 5:
            print(f"actual dt: {actual_dt}")


        self.esp.serial.reset_input_buffer()
        raw_state = self.esp.receive_state()
        while raw_state is None:
            raw_state = self.esp.receive_state()

        self.current_step += 1

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

        reward = self._calculate_reward(raw_state, act, terminated)
        
        return self._get_obs(raw_state, actual_dt), reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()

if __name__ == "__main__":
    test = CartPoleESP32Env()
    print("swing it")
    time.sleep(2)
    test.active_damp()
    print("dampened")