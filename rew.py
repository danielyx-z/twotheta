import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib import cm
import math
from esp_env import CartPoleESP32Env  # Import your class

env = CartPoleESP32Env.__new__(CartPoleESP32Env)

# 2. Manually set the 1 variable the reward function actually uses
# (Since __init__ didn't run, we have to provide this)
env.max_pos = 31900.0
def get_env_reward(t1, v1):
    # Dummy values for the parts of the state we don't care about for this plot
    # state = [t1, t2, v1, v2, pos]
    dummy_state = [t1, 0.0, v1, 0.0, 0.0] 
    dummy_action = np.array([0.0])
    terminated = False
    
    # Call the protected method from your class
    return env._calculate_reward(dummy_state, dummy_action, dummy_action, terminated)

# 2. Create meshgrid for Angle and Velocity
angle_range = np.linspace(0, 2*np.pi, 200)
velocity_range = np.linspace(-10, 10, 200)
T1, V1 = np.meshgrid(angle_range, velocity_range)

# 3. Vectorize and Calculate
R = np.vectorize(get_env_reward)(T1, V1)

# 4. Plotting
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(T1, V1, R, cmap=cm.magma, linewidth=0, antialiased=True, alpha=0.9)

# Labels
ax.set_xlabel('Angle (radians)')
ax.set_ylabel('Velocity (rad/s)')
ax.set_zlabel('Reward')


# Set Ticks to Pi
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.show()