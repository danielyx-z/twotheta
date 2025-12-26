import time
import numpy as np
from esp_controller import ESP32SerialController
import math

def energy_damping(port="COM8"):
    esp = ESP32SerialController(port, 921600)
    
    # TUNING
    K_ENERGY = 0.2   # How fast to suck energy out. Start small.
    K_CENTER = 0.001    # Pull back to center
    MAX_POS = 30000.0

    try:
        while True:
            state = esp.receive_state()
            if not state: continue
            
            t1, v1, pos = state[0], state[2], state[4]

            # 1. Energy-Based Action
            # Energy damping is: u = k * (E_current - E_target) * cos(theta) * v1
            # But we can simplify to: u = K * v1 * cos(theta)
            # This works because cos(theta) is negative when hanging down, 
            # flipping the damping direction automatically.
            u_energy = K_ENERGY * v1 * math.cos(t1)

            # 2. Stay on Rail
            u_center = -K_CENTER * (pos / MAX_POS)

            # 3. Final Command
            action = -np.clip(u_energy + u_center, -0.8, 0.8)

            # 4. Hard Stop Deadzone
            if abs(v1) < 0.2 and abs(t1) < 0.5:
                action = u_center

            esp.move(float(action))
            print(f"Energy Action: {u_energy:>5.2f} | T1: {t1:>5.2f}", end="\r")
            time.sleep(0.2)
    except KeyboardInterrupt:
        esp.move(0.0)

energy_damping()