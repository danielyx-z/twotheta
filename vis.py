import pygame
import numpy as np
import math
import struct
from multiprocessing import shared_memory
from collections import deque
import sys
import ctypes
import os

class CartPoleVisualizer:
    def __init__(self, transparency=180):
        pygame.init()
        self.width = 1600
        self.height = 1000
        
        # Enable Resizing and No Frame
        # Note: On some systems, NOFRAME prevents resizing. Use RESIZABLE.
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
        pygame.display.set_caption("CartPole RL Debugger (Press 'T' to toggle Click-Through)")
        
        self.transparency = transparency
        self.click_through = True
        self.setup_window_styles()
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.small_font = pygame.font.Font(None, 22)
        
        self.BG = (15, 15, 20)
        self.WHITE = (255, 255, 255)
        self.RED = (255, 80, 80)
        self.GREEN = (80, 255, 120)
        self.BLUE = (100, 150, 255)
        self.YELLOW = (255, 220, 80)
        self.GRAY = (100, 100, 110)
        self.DARK_GRAY = (40, 40, 45)
        
        self.max_history = 300
        self.reward_history = deque(maxlen=self.max_history)
        self.angle_history = deque(maxlen=self.max_history)
        self.position_history = deque(maxlen=self.max_history)
        self.action_history = deque(maxlen=self.max_history)
        self.velocity_history = deque(maxlen=self.max_history)
        
        try:
            self.shm = shared_memory.SharedMemory(name="cartpole_viz", create=False, size=72)
            print("Connected to shared memory!")
        except FileNotFoundError:
            print("ERROR: Shared memory not found.")
            self.shm = None
        
        self.running = True

    def setup_window_styles(self):
        """Initializes Windows-specific transparency and click-through"""
        if sys.platform == 'win32':
            self.hwnd = pygame.display.get_wm_info()['window']
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            
            # Set layered style
            style = ctypes.windll.user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED)
            self.update_styles()

    def update_styles(self):
        """Updates transparency and click-through state dynamically"""
        if sys.platform == 'win32':
            GWL_EXSTYLE = -20
            WS_EX_TRANSPARENT = 0x00000020
            
            style = ctypes.windll.user32.GetWindowLongW(self.hwnd, GWL_EXSTYLE)
            if self.click_through:
                style |= WS_EX_TRANSPARENT
            else:
                style &= ~WS_EX_TRANSPARENT
            
            ctypes.windll.user32.SetWindowLongW(self.hwnd, GWL_EXSTYLE, style)
            ctypes.windll.user32.SetLayeredWindowAttributes(self.hwnd, 0, self.transparency, 0x2)

    def read_data(self):
        if self.shm is None: return None
        try:
            data = struct.unpack('15f3i', bytes(self.shm.buf[:72]))
            return {
                'obs': np.array(data[0:6]), 'raw_state': np.array(data[6:12]),
                'action': data[12], 'reward': data[13], 'dt': data[14],
                'step': data[15], 'terminated': bool(data[16]), 'truncated': bool(data[17])
            }
        except: return None

    def draw_pendulum(self, data):
        # Use relative positioning based on current width/height
        center_x = self.width // 6
        center_y = self.height // 2
        
        pygame.draw.rect(self.screen, self.DARK_GRAY, (10, 40, 400, 500), border_radius=10)
        title = self.font.render("Physical State", True, self.WHITE)
        self.screen.blit(title, (center_x - 60, 60))
        
        theta, pos = data['raw_state'][0], data['raw_state'][4]
        track_width, pole_length = 300, 150
        track_y = center_y + 50
        
        # Cart & Pole
        cart_x = center_x + (pos / 31900.0) * (track_width//2)
        pygame.draw.line(self.screen, self.GRAY, (center_x-150, track_y), (center_x+150, track_y), 4)
        pygame.draw.rect(self.screen, self.BLUE, (cart_x-15, track_y-20, 30, 20))
        
        angle_up = theta - math.pi
        px = cart_x + pole_length * math.sin(angle_up)
        py = (track_y-20) - pole_length * math.cos(angle_up)
        pygame.draw.line(self.screen, self.RED, (cart_x, track_y-20), (px, py), 6)
        pygame.draw.circle(self.screen, self.YELLOW, (int(px), int(py)), 10)

    def draw_plots(self, data):
        self.reward_history.append(data['reward'])
        self.angle_history.append(data['raw_state'][0])
        self.action_history.append(data['action'])
        
        # Anchor plots to the right side of the window
        plot_width = min(600, self.width // 2)
        plot_x = self.width - plot_width - 20
        plot_height = (self.height - 150) // 3
        
        plots = [
            ("Reward", self.reward_history, self.GREEN, -1, 1),
            ("Angle", self.angle_history, self.RED, 0, 2*math.pi),
            ("Action", self.action_history, self.YELLOW, -1, 1)
        ]
        
        for i, (title, history, color, y_min, y_max) in enumerate(plots):
            y_off = 60 + i * (plot_height + 20)
            rect = pygame.Rect(plot_x, y_off, plot_width, plot_height)
            pygame.draw.rect(self.screen, self.DARK_GRAY, rect)
            
            if len(history) > 1:
                pts = []
                for j, v in enumerate(history):
                    tx = plot_x + (j / (self.max_history-1)) * plot_width
                    norm = np.clip((v - y_min) / (y_max - y_min), 0, 1)
                    ty = y_off + plot_height - (norm * plot_height)
                    pts.append((tx, ty))
                pygame.draw.lines(self.screen, color, False, pts, 2)
            
            t_surf = self.small_font.render(title, True, self.WHITE)
            self.screen.blit(t_surf, (plot_x, y_off - 20))

    def run(self):
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.w, event.h
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_t:
                        self.click_through = not self.click_through
                        self.update_styles()
                        print(f"Click-through: {self.click_through}")
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        self.transparency = min(255, self.transparency + 20)
                        self.update_styles()
                    elif event.key == pygame.K_MINUS:
                        self.transparency = max(50, self.transparency - 20)
                        self.update_styles()

            data = self.read_data()
            self.screen.fill(self.BG)
            if data:
                self.draw_pendulum(data)
                self.draw_plots(data)
            
            status = "LOCKED (Click-through)" if self.click_through else "EDIT MODE (Drag edges to crop)"
            hint = self.small_font.render(f"{status} | T: Toggle Lock | +/-: Alpha | ESC: Exit", True, self.GRAY)
            self.screen.blit(hint, (10, self.height - 25))
            
            pygame.display.flip()
            self.clock.tick(60)
        pygame.quit()

if __name__ == "__main__":
    viz = CartPoleVisualizer(transparency=180)
    viz.run()