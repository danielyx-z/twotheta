import gym
from gym import spaces
import numpy as np
from esp32_udp import ESP32Controller
import time

class Cartpole(gym.Env):
    def __init__(self, esp_ip, target_angles=(0.0, 0.0)):
        super().__init__()
        self.esp = ESP32Controller(esp_ip)
        self.dt = 0.02
        self.target_angles = np.array(target_angles)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.prev_angles = None

    def reset(self):
        angles, ang_vel = self.esp.get_angle()
        self.prev_angles = angles
        return self._get_obs(angles, ang_vel)

    def _get_obs(self, angles, ang_vel):
        theta1, theta2 = angles
        obs = np.array([
            np.sin(theta1), np.cos(theta1),
            np.sin(theta2), np.cos(theta2),
            ang_vel[0], ang_vel[1]
        ], dtype=np.float32)
        return obs

    def _reward(self, angles):
        return -(2.0 * (angles[0] - self.target_angles[0])**2 + 1.0 * (angles[1] - self.target_angles[1])**2)

    def step(self, action):
        self.esp.move(float(action[0]))
        time.sleep(self.dt)
        angles, ang_vel = self.esp.get_angle()
        obs = self._get_obs(angles, ang_vel)
        reward = self._reward(angles)
        done = False
        info = {}
        return obs, reward, done, info

    def close(self):
        self.esp.move(0)
        self.esp.close()
