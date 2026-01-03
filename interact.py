import sys
import time
import math
import pygame
from esp_controller import ESP32SerialController

# Configuration
SERIAL_PORT = "/dev/ttyUSB0" 
BAUD_RATE = 921600
FPS = 120
L1 = 120  
L2 = 150  
POS_SCALE = 0.01  

class PendulumVisualizer:
    def __init__(self, port, baudrate):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Double Pendulum Cartpole")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True
        
        # Initialize our modular controller
        self.esp = ESP32SerialController(port, baudrate)
        
        self.last_state = None
        self.last_update_time = time.time()
        self.ms_since_update = 0
        self.trail_points = []
        self.max_trail_length = 100

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        self.screen.blit(img, (x, y))

    def run(self):
        base_y = 200
        center_x = 400
        
        while self.running:
            # 1. Event Handling
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q: self.running = False
                    elif ev.key == pygame.K_LEFT: self.esp.move(-0.01)
                    elif ev.key == pygame.K_RIGHT: self.esp.move(1.0)
                    elif ev.key == pygame.K_SPACE: self.esp.move(0.0)
                    elif ev.key == pygame.K_h: self.esp.move(10)
            # 2. Data Acquisition
            latest_state = self.esp.receive_state()
            buffer_backlog = self.esp.get_buffer_size()
            
            if latest_state:
                self.last_state = latest_state
                self.last_update_time = time.time()

            self.ms_since_update = (time.time() - self.last_update_time) * 1000
            
            # 3. Rendering
            self.screen.fill((0, 0, 0))

            if self.last_state is None:
                self.draw_text("Waiting for ESP32 data...", 300, 280)
                pygame.display.flip()
                self.clock.tick(FPS)
                continue

            theta1, theta2, vel1, vel2, pos, motor_speed = self.last_state
            
            # Kinematics
            base_x = center_x + pos * POS_SCALE
            x1 = base_x - L1 * math.sin(theta1)
            y1 = base_y + L1 * math.cos(theta1)
            x2 = x1 - L2 * math.sin(theta1 + theta2)
            y2 = y1 + L2 * math.cos(theta1 + theta2)

            # Trail Logic
            self.trail_points.append((int(x2), int(y2)))
            if len(self.trail_points) > self.max_trail_length:
                self.trail_points.pop(0)

            # Draw Trail
            for i, (px, py) in enumerate(self.trail_points):
                alpha = int(255 * (i / len(self.trail_points)))
                size = (2 + 6 * (i / len(self.trail_points))) / 2
                glow_surf = pygame.Surface((size*4, size*4), pygame.SRCALPHA)
                pygame.draw.circle(glow_surf, (0, 150, 255, alpha//3), (size*2, size*2), size*2)
                self.screen.blit(glow_surf, (px - size*2, py - size*2))

            # Draw Arms and Joints
            pygame.draw.line(self.screen, (200, 200, 200), (base_x, base_y), (x1, y1), 6)
            pygame.draw.line(self.screen, (100, 200, 100), (x1, y1), (x2, y2), 6)
            pygame.draw.circle(self.screen, (180, 0, 0), (int(x1), int(y1)), 8)
            pygame.draw.circle(self.screen, (0, 0, 200), (int(x2), int(y2)), 8)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(base_x), int(base_y)), 6)

            # UI Overlay
            latency_color = (0, 255, 0) if self.ms_since_update < 50 else (255, 0, 0)
            self.draw_text(f"Last Update: {int(self.ms_since_update)} ms ago", 10, 10, latency_color)
            self.draw_text(f"Buffer Backlog: {buffer_backlog} bytes", 10, 35)
            self.draw_text(f"Pos: {pos:+.1f}", 10, 60)
            self.draw_text(f"θ1: {math.degrees(theta1):+7.2f}° | ω1: {math.degrees(vel1):+7.2f}°/s", 10, 85)
            self.draw_text(f"θ2: {math.degrees(theta2):+7.2f}° | ω2: {math.degrees(vel2):+7.2f}°/s", 10, 110)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.esp.close()
        pygame.quit()

if __name__ == "__main__":
    vis = PendulumVisualizer(SERIAL_PORT, BAUD_RATE)
    vis.run()