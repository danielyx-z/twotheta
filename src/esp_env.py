import gymnasium as gym
import numpy as np
from gymnasium import spaces
from esp_controller import ESP32SerialController
import math
import time
from multiprocessing import shared_memory
import struct

class CartPoleESP32Env(gym.Env):
    def __init__(self, port="COM8", baudrate=921600, max_steps=1000, enable_viz=False):
        super().__init__()
        self.esp = ESP32SerialController(port, baudrate)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        
        self.max_pos = 31900.0  
        self.max_motor_speed = 90000.0
        self.max_episode_steps = max_steps
        self.current_step = 0
        self.overspeed_counter = 0
        self.MAX_OVERSPEED_FRAMES = 20
        self.SPIN_THRESHOLD = 3 * math.pi 
        self.last_step_time = time.perf_counter()
        self.first_step_of_episode = True
        
        # Visualization setup
        self.enable_viz = enable_viz
        self.shm = None
        if self.enable_viz:
            self._setup_shared_memory()

    def _setup_shared_memory(self):
        """Setup shared memory for visualization data"""
        # Data format: 
        # 6 floats (obs) + 6 floats (raw_state) + 1 float (action) + 1 float (reward) 
        # + 1 float (dt) + 1 int (step) + 1 int (terminated) + 1 int (truncated)
        # Total: 15 floats + 3 ints = 60 + 12 = 72 bytes
        try:
            self.shm = shared_memory.SharedMemory(name="cartpole_viz", create=True, size=72)
        except FileExistsError:
            self.shm = shared_memory.SharedMemory(name="cartpole_viz", create=False, size=72)
        
        # Initialize with zeros
        data = struct.pack('15f3i', *([0.0]*15 + [0]*3))
        self.shm.buf[:72] = data

    def _update_viz_data(self, obs, raw_state, action, reward, terminated, truncated, dt):
        """Update shared memory with current step data"""
        if not self.enable_viz or self.shm is None:
            return
        
        # Convert raw_state to list if it's a tuple or array
        if isinstance(raw_state, (tuple, list)):
            raw_state_list = list(raw_state)
        else:
            raw_state_list = raw_state.tolist()
        
        data = struct.pack('15f3i',
            *obs.tolist(),  # 6 floats
            *raw_state_list,  # 6 floats
            float(action),  # 1 float
            float(reward),  # 1 float
            float(dt),  # 1 float
            self.current_step,  # 1 int
            int(terminated),  # 1 int
            int(truncated)  # 1 int
        )
        self.shm.buf[:72] = data

    def _get_obs(self, state, dt_measured):
        t1, t2, v1, v2, pos, motor_vel = state
        return np.array([
            math.sin(t1), 
            math.cos(t1),
            v1 / self.SPIN_THRESHOLD, 
            pos / self.max_pos,
            motor_vel / self.max_motor_speed,
            dt_measured * 60
        ], dtype=np.float32)

    def _calculate_reward(self, state, action, terminated):
        t1, v1, pos = state[0], state[2], state[4]

        if terminated:
            return -30.0 

        error = (t1 - math.pi + math.pi) % (2 * math.pi) - math.pi
        uprightness = ((-math.cos(t1) + 1) / 2) ** 2

        r_base = (-math.cos(t1) + 1) / 2

        r_pos = -0.1 * abs(pos / self.max_pos) ** 2
        
        r_velocity = -0.02 * uprightness * (v1 ** 2)
        
        r_action = -0.01 * (action ** 2)

        return float(r_base + r_pos + r_velocity + r_action)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.esp.serial.reset_output_buffer()
        self.first_step_of_episode = True
        self.esp.move(10.0) 
        self.last_step_time = time.perf_counter()

        obs = self._get_obs(np.zeros(6, dtype=np.float32), 0.018)
        
        if self.enable_viz:
            self._update_viz_data(obs, np.zeros(6, dtype=np.float32), 0.0, 0.0, False, False, 0.02)
        
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
                state = self.esp.receive_state()

            t1, v1, pos = state[0], state[2], state[4]
            u_energy = 0.2 * v1 * math.cos(t1)
            u_center = 0.01 * (pos / self.max_pos)
            action = -np.clip(u_energy + u_center, -0.8, 0.8)

            if abs(v1) < 0.3 and math.cos(t1) > 0.97:
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
        current_action = np.clip(action[0], -1.0, 1.0)

        if self.first_step_of_episode:
            self.esp.serial.reset_input_buffer()
            raw_state = self.esp.receive_state()
            while raw_state is None:
                raw_state = self.esp.receive_state()
            self.first_step_of_episode = False

        self.last_step_time = time.perf_counter()
        self.esp.move(float(current_action))

        #self.esp.serial.reset_input_buffer()
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

        reward = self._calculate_reward(raw_state, current_action, terminated)
        
        current_time = time.perf_counter()
        actual_dt = current_time - self.last_step_time
        obs = self._get_obs(raw_state, actual_dt)
        self.last_step_time = current_time

        # Update visualization data
        if self.enable_viz:
            self._update_viz_data(obs, raw_state, current_action, reward, terminated, truncated, actual_dt)

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.esp.move(0.0) 
        self.esp.close()
        
        if self.shm is not None:
            self.shm.close()
            try:
                self.shm.unlink()
            except:
                pass

if __name__ == "__main__":
    test = CartPoleESP32Env(enable_viz=True)
    print("swing it")
    time.sleep(2)
    test.active_damp()
    print("dampened")