import numpy as np
import math
import time
from esp_controller import ESP32SerialController

def wrap_angle_pi(x):
    return (x + math.pi) % (2 * math.pi) - math.pi

class PIDController:
    def __init__(self, kp, kd, output_limit=1.0, d_alpha=0.2):
        self.kp = kp
        self.kd = kd
        self.output_limit = output_limit
        self.d_alpha = d_alpha
        self.prev_error = 0.0
        self.prev_time = None
        self.d_filt = 0.0

    def reset(self):
        self.prev_error = 0.0
        self.prev_time = None
        self.d_filt = 0.0

    def compute(self, error, t):
        if self.prev_time is None:
            self.prev_time = t
            self.prev_error = error
            return 0.0
        dt = t - self.prev_time
        if dt <= 0: return 0.0
        d_raw = (error - self.prev_error) / dt
        self.d_filt = (self.d_alpha * d_raw + (1.0 - self.d_alpha) * self.d_filt)
        u = self.kp * error + self.kd * self.d_filt
        self.prev_error = error
        self.prev_time = t
        return np.clip(u, -self.output_limit, self.output_limit)

def balance_pendulum(port="COM8", baudrate=921600, duration=60):
    esp = ESP32SerialController(port, baudrate)

    # Balancing PID
    angle_pid = PIDController(kp=10.0, kd=0.2, output_limit=1.0, d_alpha=0.05)

    # --- SIMPLE SWING-UP PARAMETERS ---
    k_swing = 0.15 # Strength of push
    activation_half_width = math.radians(25.0) 

    target_angle = math.pi - 0.008
    start_time = time.time()

    try:
        while time.time() - start_time < duration:
            now = time.time()
            state = esp.receive_state()
            if state is None: continue

            t1, t2, v1, v2, pos = state
            angle_err = wrap_angle_pi(target_angle - t1)

            # Safety: Cut if spinning like crazy
            if abs(v1) > 15:
                esp.move(0.0)
                continue

            if abs(angle_err) <= activation_half_width:
                u = -angle_pid.compute(angle_err, now)
                mode = "BALANCING"
            else:
                angle_pid.reset()
                taper = max(0, math.cos(t1)-0.1)  / 2
                
                u = k_swing * v1 * taper
                mode = "SWING-UP "
            
            u = np.clip(u, -1.0, 1.0)
            esp.move(float(u))

            print(f"θ={math.degrees(t1):6.1f}° | Mode: {mode} | u={u:6.3f}", end="\r")
            time.sleep(0.001)

    except KeyboardInterrupt:
        pass
    finally:
        esp.move(0.0)
        esp.close()

if __name__ == "__main__":
    balance_pendulum()