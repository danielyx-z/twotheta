import numpy as np
import math
import time
from esp_controller import ESP32SerialController

def wrap_angle_pi(x):
    """Normalizes angle to range [-pi, pi]"""
    return (x + math.pi) % (2 * math.pi) - math.pi

class SimplePD:
    def __init__(self, kp, kd, max_speed=1.0):
        self.kp = kp
        self.kd = kd
        self.max_speed = max_speed
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, error, t):
        if self.prev_time is None:
            self.prev_time = t
            self.prev_error = error
            return 0.0
        
        dt = t - self.prev_time
        if dt <= 0.0001: return 0.0 # Prevent divide by zero

        # Derivative (D): Rate of change of error
        # High Kd acts like a "damper" to stop oscillation
        derivative = (error - self.prev_error) / dt

        # Control Law: u = Kp * e + Kd * de/dt
        output = (self.kp * error) + (self.kd * derivative)
        
        self.prev_error = error
        self.prev_time = t
        
        return np.clip(output, -self.max_speed, self.max_speed)

def balance_pendulum(port="/dev/ttyUSB0", baudrate=921600):
    esp = ESP32SerialController(port, baudrate)

    # TUNING GUIDE:
    # 1. Start with Kp=0, Kd=0.
    # 2. Increase Kp until the cart tries to "catch" the pole but oscillates forever.
    # 3. Increase Kd until the oscillations dampen out and it becomes sticky/stiff.
    # Recommended Start: Kp=15.0, Kd=0.5 (Adjust Kd in small steps like 0.1)
    pid = SimplePD(kp=25.0, kd=0.8, max_speed=1.0)

    # Safety limits
    activation_limit = math.radians(20.0) # Only balance inside +/- 20 deg
    target_angle = math.pi # Upright position

    print("Ready to balance. Hold pole upright...")

    try:
        while True:
            state = esp.receive_state()
            if state is None: continue

            # Unpack state (theta, omega, cart_pos, cart_vel, etc.)
            # Assuming t1 is angle, v1 is angular velocity (if available)
            t1, _, _, _, _, _ = state 
            now = time.time()

            # Calculate error (0 is upright)
            angle_err = wrap_angle_pi(target_angle - t1)

            # --- CONTROL LOGIC ---
            if abs(angle_err) < activation_limit:
                # We are in the "Balance Zone"
                # Note: The sign (-) depends on your motor wiring. 
                # If it runs AWAY from the fall, remove the minus.
                u = -pid.compute(angle_err, now)
                
                print(f"BALANCING | Err: {math.degrees(angle_err):.1f}° | Cmd: {u:.2f}", end="\r")
            else:
                # We are falling/fallen -> Cut motors for safety
                u = 0.0
                pid.prev_time = None # Reset PID timer so it doesn't jerk on restart
                print(f"IDLE      | Err: {math.degrees(angle_err):.1f}° | Cmd: 0.00", end="\r")

            esp.move(float(u))

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        esp.move(0.0)
        esp.close()

if __name__ == "__main__":
    balance_pendulum()