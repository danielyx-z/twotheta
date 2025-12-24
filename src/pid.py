import time
import math
import numpy as np
import keyboard
from esp_env import CartPoleESP32Env

# INITIAL GAINS
params = {
    'KP': 5.0,
    'KI': 0.01,
    'KD': 0.3,
    'ALPHA': 0.6,    
    'FF': 0.05       
}

# 120 degrees total (60 degrees each way from vertical)
# 60 * (pi / 180) = ~1.047 radians
BALANCING_THRESHOLD = 1.047 

STEP = {'KP': 0.1, 'KI': 0.001, 'KD': 0.01, 'ALPHA': 0.02, 'FF': 0.01}

def run_pid_balancer():
    env = CartPoleESP32Env(port="COM7", baudrate=921600) 
    print("\n--- UPRIGHT ONLY BALANCER (120° Range) ---")
    
    obs, _ = env.reset()
    prev_error = 0
    integral = 0
    last_time = time.perf_counter()
    target_angle = math.pi 

    try:
        while True:
            # Handle Keyboard Tuning
            if keyboard.is_pressed('q'): params['KP'] += STEP['KP']
            if keyboard.is_pressed('a'): params['KP'] -= STEP['KP']
            if keyboard.is_pressed('w'): params['KI'] += STEP['KI']
            if keyboard.is_pressed('s'): params['KI'] -= STEP['KI']
            if keyboard.is_pressed('e'): params['KD'] += STEP['KD']
            if keyboard.is_pressed('d'): params['KD'] -= STEP['KD']
            if keyboard.is_pressed('r'): params['ALPHA'] += STEP['ALPHA']
            if keyboard.is_pressed('f'): params['ALPHA'] -= STEP['ALPHA']
            if keyboard.is_pressed('esc'): break

            now = time.perf_counter()
            dt = now - last_time
            last_time = now
            if dt <= 0: dt = 0.001

            s1, c1 = obs[0], obs[1]
            current_angle = math.atan2(s1, c1)

            # Error calculation (Shortest path to PI)
            raw_error = (target_angle - current_angle + math.pi) % (2 * math.pi) - math.pi
            
            # --- ACTIVATION CHECK ---
            # If the absolute error is greater than 60 degrees, don't move
            if abs(raw_error) > BALANCING_THRESHOLD:
                action = np.array([0.0])
                integral = 0 # Reset integral so it doesn't "jump" when caught
                prev_error = 0
                status_mode = "INACTIVE"
            else:
                # Normal PID Logic
                shaped_error = np.sign(raw_error) * (abs(raw_error) ** params['ALPHA'])
                
                integral += shaped_error * dt
                integral = np.clip(integral, -0.4, 0.4)
                derivative = (shaped_error - prev_error) / dt
                
                output = (params['KP'] * shaped_error) + (params['KI'] * integral) + (params['KD'] * derivative)
                
                # Friction floor
                if abs(output) > 0.001:
                    output += np.sign(output) * params['FF']
                
                action = np.clip([output], -1, 1)
                prev_error = shaped_error
                status_mode = "ACTIVE  "

            obs, _, _, _, _ = env.step(action * -1)

            print(f"\r[{status_mode}] P:{params['KP']:4.1f} α:{params['ALPHA']:4.2f} | Err:{raw_error:6.3f} | Act:{action[0]:5.2f}", end="")
            time.sleep(0.001)

    finally:
        env.close()

if __name__ == "__main__":
    run_pid_balancer()