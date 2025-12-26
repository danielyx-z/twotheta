import numpy as np
import math

class VelocityPID:
    def __init__(self, kp, ki, kd, max_speed=1.0, d_alpha=0.2, i_limit=0.2):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_speed = max_speed
        self.i_limit = i_limit
        self.d_alpha = d_alpha
        
        self.prev_error = 0.0
        self.prev_time = None
        self.integral = 0.0
        self.d_filt = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.prev_time = None
        self.integral = 0.0
        self.d_filt = 0.0

    def compute(self, error, t):
        if self.prev_time is None:
            self.prev_time = t
            self.prev_error = error
            return 0.0
        
        dt = t - self.prev_time
        if dt <= 0: return 0.0

        # Integral
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        
        # Derivative (Low pass filtered)
        d_raw = (error - self.prev_error) / dt
        self.d_filt = (self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt)
        
        # PID Calc
        v_cmd = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.d_filt)
        
        self.prev_error, self.prev_time = error, t
        return np.clip(v_cmd, -self.max_speed, self.max_speed)