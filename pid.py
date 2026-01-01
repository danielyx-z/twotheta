import numpy as np
import math
import time
from esp_controller import ESP32SerialController

def wrap_angle_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

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

        # 1. Integral: Handles the 'Lean' (very small limit for velocity)
        self.integral = np.clip(self.integral + error * dt, -self.i_limit, self.i_limit)
        
        # 2. Derivative: The 'Damping' (critical for velocity control)
        d_raw = (error - self.prev_error) / dt
        self.d_filt = (self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt)
        
        # 3. Velocity Command
        # Notice: we don't need a massive Kp here because speed is powerful
        v_cmd = (self.kp * error) + (self.ki * self.integral) + (self.kd * self.d_filt)
        
        self.prev_error, self.prev_time = error, t
        return np.clip(v_cmd, -self.max_speed, self.max_speed)

def balance_pendulum(port="COM8", baudrate=921600):
    esp = ESP32SerialController(port, baudrate)

    # VELOCITY-SPECIFIC TUNING:
    # Kp is lower (4.0) to prevent aggressive over-correction.
    # Kd is higher (1.8) to provide stiff damping.
    # Ki is low (1.0) just to find that vertical offset.
    pid = VelocityPID(kp=30.0, ki=1, kd=4, max_speed=1.0)

    activation_half_width = math.radians(20.0)
    target_angle = math.pi - 0.0275 # Use pi; the I-term handles the rest.

    try:
        while True:
            now = time.time()
            state = esp.receive_state()
            if state is None: continue

            t1, _, v1, _, _ = state
            angle_err = wrap_angle_pi(target_angle - t1)

            if abs(angle_err) <= activation_half_width:
                # In velocity control, we usually move TOWARD the lean to catch it.
                # If your cart moves the wrong way, remove the minus sign here.
                u = -pid.compute(angle_err, now)
                mode = "BALANCE"
            else:
                pid.reset()
                # Swing up: Output is also a velocity
                u = 0.25 * v1 * (max(0, math.cos(t1)-0.1))
                mode = "SWINGUP"

            esp.move(float(u))
            print(f"M: {mode} | Err: {math.degrees(angle_err):.1f}° | V_cmd: {u:.2f}", end="\r")

    except KeyboardInterrupt:
        pass
    finally:
        esp.move(0.0)
        esp.close()

if __name__ == "__main__":
    balance_pendulum()