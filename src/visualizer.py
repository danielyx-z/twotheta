import sys
import time
import math
import pygame
import serial
import struct 

SERIAL_PORT = "COM7" 
BAUD_RATE = 921600
FPS = 120
L1 = 120  
L2 = 150  
POS_SCALE = 0.01  

STATE_HEADER = b'\xAA\x55'  
CMD_HEADER = b'\x55\xAA'    
STATE_PACKET_SIZE = 22  
CMD_PACKET_SIZE = 6     

class ESP32SerialController:
    def __init__(self, port, baudrate):
        self.serial = None
        try:
            self.serial = serial.Serial(port, baudrate, timeout=0)
            time.sleep(0.5)  
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except serial.SerialException as e:
            sys.exit(1)

    def move(self, action):
        if self.serial and self.serial.is_open:
            packet = CMD_HEADER + struct.pack('f', action)
            self.serial.write(packet)

    def receive_state(self):
        if not (self.serial and self.serial.is_open):
            return None
        
        if self.serial.in_waiting < STATE_PACKET_SIZE:
            return None
        
        max_search = 200  
        search_count = 0
        
        while self.serial.in_waiting >= STATE_PACKET_SIZE and search_count < max_search:
            first_byte = self.serial.read(1)
            if len(first_byte) == 0: break
            
            if first_byte[0] == 0xAA:
                second_byte = self.serial.read(1)
                if len(second_byte) == 0: break
                
                if second_byte[0] == 0x55:
                    data = self.serial.read(20)
                    if len(data) == 20:
                        try:
                            state = struct.unpack('5f', data)
                            if all(math.isfinite(x) for x in state):
                                while self.serial.in_waiting >= STATE_PACKET_SIZE:
                                    peek1 = self.serial.read(1)
                                    if len(peek1) > 0 and peek1[0] == 0xAA:
                                        peek2 = self.serial.read(1)
                                        if len(peek2) > 0 and peek2[0] == 0x55:
                                            data = self.serial.read(20)
                                            if len(data) == 20:
                                                try:
                                                    new_state = struct.unpack('5f', data)
                                                    if all(math.isfinite(x) for x in new_state):
                                                        state = new_state
                                                except struct.error: pass
                                return state
                        except struct.error: pass
            search_count += 1
        return None

    def get_buffer_size(self):
        return self.serial.in_waiting if self.serial else 0

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()

class PendulumVisualizer:
    def __init__(self, port, baudrate):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Double Pendulum Cartpole")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True
        self.esp = ESP32SerialController(port, baudrate)
        self.last_state = None
        self.last_update_time = time.time()
        self.ms_since_update = 0

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        self.screen.blit(img, (x, y))

    def run(self):
        base_y = 200
        center_x = 400
        
        while self.running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q: self.running = False
                    elif ev.key == pygame.K_LEFT: self.esp.move(-0.6)
                    elif ev.key == pygame.K_RIGHT: self.esp.move(0.6)
                    elif ev.key == pygame.K_SPACE: self.esp.move(0.0)

            latest_state = self.esp.receive_state()
            buffer_backlog = self.esp.get_buffer_size()
            
            if latest_state:
                self.last_state = latest_state
                self.last_update_time = time.time()

            self.ms_since_update = (time.time() - self.last_update_time) * 1000
            self.screen.fill((0, 0, 0))

            if self.last_state is None:
                self.draw_text("Waiting for ESP32 data...", 300, 280)
                pygame.display.flip()
                self.clock.tick(FPS)
                continue

            theta2, theta1, vel1, vel2, pos = self.last_state
            base_x = center_x + pos * POS_SCALE
            x1 = base_x - L1 * math.sin(theta1)
            y1 = base_y + L1 * math.cos(theta1)
            x2 = x1 - L2 * math.sin(theta1 + theta2)
            y2 = y1 + L2 * math.cos(theta1 + theta2)

            pygame.draw.line(self.screen, (200, 200, 200), (base_x, base_y), (x1, y1), 6)
            pygame.draw.line(self.screen, (100, 200, 100), (x1, y1), (x2, y2), 6)
            pygame.draw.circle(self.screen, (180, 0, 0), (int(x1), int(y1)), 8)
            pygame.draw.circle(self.screen, (0, 0, 200), (int(x2), int(y2)), 8)
            pygame.draw.circle(self.screen, (255, 255, 0), (int(base_x), int(base_y)), 6)

            self.draw_text(f"Last Update: {int(self.ms_since_update)} ms ago", 10, 10, (0, 255, 0) if self.ms_since_update < 50 else (255, 0, 0))
            self.draw_text(f"Buffer Backlog: {buffer_backlog} bytes", 10, 35)
            self.draw_text(f"Pos: {pos:+.1f}", 10, 60)

            pygame.display.flip()
            self.clock.tick(FPS)

        self.esp.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    vis = PendulumVisualizer(SERIAL_PORT, BAUD_RATE)
    vis.run()