import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import math


import math

def _calculate_reward(t1, v1):

    # --- 1. The Angle Reward (The Driver) ---
    # Since Top is PI, valid range is likely 0 to 2*pi or -pi to pi.
    # cos(pi) = -1. We want to MAXIMIZE reward at pi.
    # So we flip the sign of cosine.
    # Top (pi) -> -(-1) = +1.0
    # Bottom (0) -> -(1) = -1.0
    r_angle = -math.cos(t1)

    # --- 2. The Velocity Penalty (The Brake) ---
    # This acts as a damper. It is always negative.
    # If the pole spins (helicopters), v1 becomes huge, and this penalty
    # overpowers the r_angle reward, making the total reward negative.
    # The only way to get a positive net score is to be at PI (r_angle=+1) 
    # with velocity near ZERO (r_velocity approx 0).
    r_velocity = -0.05 * (v1 ** 2)

    # --- 3. Stability Bonus (The Magnet) ---
    # A sparse reward to "lock" it in at the top.
    # We need to calculate the distance from PI carefully.
    # This handles wrapping (e.g. if it goes 3.15 vs 3.13)
    error_angle = abs(t1 - math.pi)
    
    # Optional: Normalizing error if your sensor wraps weirdly (e.g. 0 to 2pi)
    # error_angle = abs(math.atan2(math.sin(t1-math.pi), math.cos(t1-math.pi)))

    r_stability = 0.0
    # If within ~11 degrees of top (0.2 rad) and moving slowly
    if error_angle < 0.2 and abs(v1) < 0.5:
        r_stability = 2.0  # Big bonus for holding the balance


    # Total
    reward = r_angle + r_velocity + r_stability 
    return float(reward)

# Reward calculation function
import math

def calculate_reward(t1, v1):


    # 1. Normalized Error (Shortest distance to pi)
    error = (t1 - math.pi + math.pi) % (2 * math.pi) - math.pi
    
    # 2. Angle Reward (The "Mountain")
    # Base swing-up guide (-1 to +1)
    r_swingup = -math.cos(t1)
    
    # Precision bonus (The "Peak")
    # Smooth Gaussian bell curve
    width = 0.15 
    r_precision = 2.0 * math.exp(-(error ** 2) / (2 * width ** 2))

    # 3. State-Dependent Velocity Penalty (The "Brake")
    # We define 'uprightness' from 0 (bottom) to 1 (top)
    uprightness = (r_swingup + 1.0) / 2.0 
    
    # Weighting: 
    # Near bottom (uprightness ~0): Penalty is almost 0.0
    # Near top (uprightness ~1): Penalty is full 0.2
    # This allows the agent to 'whip' the pole at the bottom without losing points.
    v_penalty_weight = 0.2 * (uprightness ** 2) 
    r_velocity = -v_penalty_weight * (v1 ** 2)

    reward = r_swingup + r_precision + r_velocity

    return float(reward)

# Create meshgrid for angle and velocity
angle_range = np.linspace(0, 2*np.pi, 200)
velocity_range = np.linspace(-10, 10, 200)
T1, V1 = np.meshgrid(angle_range, velocity_range)

# Calculate reward for each point
R = np.vectorize(calculate_reward)(T1, V1)

# Create the 3D plot
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot surface
surf = ax.plot_surface(T1, V1, R, cmap=cm.viridis, 
                       linewidth=0, antialiased=True, alpha=0.8)

# Add contour lines on the bottom
ax.contour(T1, V1, R, zdir='z', offset=R.min(), cmap=cm.viridis, alpha=0.5)

# Labels and title
ax.set_xlabel('Angle (radians)', fontsize=12, labelpad=10)
ax.set_ylabel('Angular Velocity (rad/s)', fontsize=12, labelpad=10)
ax.set_zlabel('Reward', fontsize=12, labelpad=10)
ax.set_title('Pendulum Reward Function Landscape\n(Angle vs Velocity)', 
             fontsize=14, pad=20)

# Add colorbar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Mark key points
# Upright position with zero velocity (optimal)
ax.scatter([np.pi], [0], [calculate_reward(np.pi, 0)], 
           color='red', s=100, marker='*', 
           label='Upright (π, 0)')

# Down position with zero velocity (starting position)
ax.scatter([0], [0], [calculate_reward(0, 0)], 
           color='orange', s=100, marker='o', 
           label='Down (0, 0)')

# Set x-axis ticks to show key angles
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

ax.legend(loc='upper left')

# Adjust viewing angle for better visualization
ax.view_init(elev=25, azim=45)

plt.tight_layout()
plt.show()

# Create a second plot: 2D heatmap with contours
fig2, ax2 = plt.subplots(figsize=(12, 8))

# Create contour plot
contour = ax2.contourf(T1, V1, R, levels=30, cmap=cm.viridis)
contour_lines = ax2.contour(T1, V1, R, levels=15, colors='white', 
                             linewidths=0.5, alpha=0.4)

# Add labels to contour lines
ax2.clabel(contour_lines, inline=True, fontsize=8)

# Add colorbar
cbar = plt.colorbar(contour, ax=ax2)
cbar.set_label('Reward', fontsize=12)

# Mark key points
ax2.plot(np.pi, 0, 'r*', markersize=20, label='Upright (π, 0)')
ax2.plot(0, 0, 'o', color='orange', markersize=12, label='Down (0, 0)')
ax2.plot(2*np.pi, 0, 'o', color='orange', markersize=12)

# Labels and title
ax2.set_xlabel('Angle (radians)', fontsize=12)
ax2.set_ylabel('Angular Velocity (rad/s)', fontsize=12)
ax2.set_title('Pendulum Reward Function - Top View\n(Contour Map)', fontsize=14)

# Set x-axis ticks
ax2.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax2.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])

ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print some statistics
print(f"Maximum reward: {R.max():.3f} at angle={angle_range[np.unravel_index(R.argmax(), R.shape)[1]]:.3f}, velocity={velocity_range[np.unravel_index(R.argmax(), R.shape)[0]]:.3f}")
print(f"Minimum reward: {R.min():.3f}")
print(f"Reward at upright (π, 0): {calculate_reward(np.pi, 0):.3f}")
print(f"Reward at down (0, 0): {calculate_reward(0, 0):.3f}")