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
        self.is_initialized = False
        # We assume the hardware sends data every ~15ms. We don't enforce it with sleep.
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 25
        self.SPIN_THRESHOLD = 3 * math.pi #rad / s
        self.last_step_time = time.perf_counter()

    def _get_obs(self, state, dt_measured):
        t1, t2, v1, v2, pos = state
        return np.array([
            math.sin(t1), math.cos(t1),
            v1, 
            pos / self.max_pos,
            dt_measured * 60
        ], dtype=np.float32)

    def _calculate_reward(self, state, action, terminated):
        t1, t2, v1, v2, pos = state
        if terminated:
            return -20.0

        upright = (1 - math.cos(t1)) / 2
        upright = upright ** 0.8
        vel_gate = math.exp(- (v1 / 3.0) ** 2)
        reward = upright * vel_gate
        reward -= 0.3 * (pos / self.max_pos) ** 2
        reward -= 0.01 * (action ** 2)

        return float(reward)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.esp.serial.reset_output_buffer()
        if not self.is_initialized:
            self.esp.move(10.0) # Trigger hardware homing
            self.is_initialized = True

        self.esp.serial.reset_input_buffer()
        state = self.esp.receive_state()
        while state is None: 
            state = self.esp.receive_state()

        self.last_step_time = time.perf_counter()

        # Default to 0.015 on first frame to prevent NN spike
        return self._get_obs(state, 0.015), {}

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
        act = np.clip(action[0], -1.0, 1.0)
        self.esp.move(float(act))

        current_time = time.perf_counter()
        actual_dt = current_time - self.last_step_time
        self.last_step_time = current_time

        assert not 0.1 < actual_dt < 1, f"sometihng up with actual dt {actual_dt}"


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
            self.esp.move(10)
            
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