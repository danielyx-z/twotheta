import sys
import time
import math
import pygame
import serial # Import pyserial
import struct # For packing/unpacking binary data

# Configuration
SERIAL_PORT = "COM7" # !!! CHANGE THIS TO YOUR ESP32's serial port !!!
BAUD_RATE = 921600
FPS = 120
L1 = 120  # pixels
L2 = 150  # pixels
POS_SCALE = 0.01  # stepper units -> pixels

# Protocol constants
STATE_HEADER = b'\xAA\x55'  # 2-byte header for state packets
CMD_HEADER = b'\x55\xAA'    # 2-byte header for command packets
STATE_PACKET_SIZE = 22  # 2 header + 20 data bytes
CMD_PACKET_SIZE = 6     # 2 header + 4 data bytes (1 float)

class ESP32SerialController:
    def __init__(self, port, baudrate):
        self.serial = None
        try:
            self.serial = serial.Serial(port, baudrate, timeout=0.1)
            print(f"Connected to ESP32 on {port} at {baudrate} baud.")
            # Clear any initial buffer garbage
            time.sleep(0.5)  # Give ESP32 time to boot
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except serial.SerialException as e:
            print(f"Error opening serial port {port}: {e}")
            sys.exit(1)

    def move(self, action):
        if self.serial and self.serial.is_open:
            # Pack command as binary: header (2 bytes) + float (4 bytes)
            packet = CMD_HEADER + struct.pack('f', action)
            self.serial.write(packet)

    def receive_state(self):
        if not (self.serial and self.serial.is_open):
            return None
        
        # Need at least one full packet
        if self.serial.in_waiting < STATE_PACKET_SIZE:
            return None
        
        # Search for 2-byte header to synchronize
        max_search = 200  # Prevent infinite loop
        search_count = 0
        
        while self.serial.in_waiting >= STATE_PACKET_SIZE and search_count < max_search:
            # Read first byte
            first_byte = self.serial.read(1)
            
            if len(first_byte) == 0:
                break
            
            # Check if it matches first header byte
            if first_byte[0] == 0xAA:
                # Check second header byte
                second_byte = self.serial.read(1)
                
                if len(second_byte) == 0:
                    break
                
                if second_byte[0] == 0x55:
                    # Found complete header! Read the rest of the packet (20 bytes)
                    data = self.serial.read(20)
                    
                    if len(data) == 20:
                        try:
                            # Unpack 5 floats
                            state = struct.unpack('5f', data)
                            # Validate that all values are finite
                            if all(math.isfinite(x) for x in state):
                                # Drain any remaining packets in buffer to get most recent
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
                                                except struct.error:
                                                    pass
                                return state
                        except struct.error:
                            pass  # Invalid packet, keep searching
            
            search_count += 1
        
        return None

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()
            print("Serial port closed.")

class PendulumVisualizer:
    def __init__(self, port, baudrate):
        pygame.init()
        self.screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("Double Pendulum Cartpole")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.running = True
        
        # ESP32 serial connection
        self.esp = ESP32SerialController(port, baudrate)
        
        # Cache last valid state (initialized to None)
        self.last_state = None
        self.state_received = False
        
        print(f"Serial communication established.")
        print("Waiting for first valid state from ESP32...")

    def draw_text(self, text, x, y, color=(255, 255, 255)):
        img = self.font.render(text, True, color)
        self.screen.blit(img, (x, y))

    def run(self):
        base_y = 200
        center_x = 400
        
        while self.running:
            # Event handling
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self.running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_q:
                        self.running = False
                    elif ev.key == pygame.K_LEFT:
                        self.esp.move(-0.6)
                    elif ev.key == pygame.K_RIGHT:
                        self.esp.move(0.6)
                    elif ev.key == pygame.K_SPACE:
                        self.esp.move(0.0)

            # --- Real-time serial update ---
            latest_state = self.esp.receive_state() # Will return None if no full packet
            
            if latest_state:
                self.last_state = latest_state  # Cache the latest good state
                if not self.state_received:
                    print("First valid state received!")
                    self.state_received = True

            # Clear screen
            self.screen.fill((0, 0, 0))

            # If we haven't received any valid state yet, show waiting message
            if self.last_state is None:
                self.draw_text("Waiting for ESP32 data...", 300, 280)
                self.draw_text("←/→: Move | Space: Stop | Q: Quit", 10, 560)
                pygame.display.flip()
                self.clock.tick(FPS)
                continue

            # Unpack state
            theta2, theta1, vel1, vel2, pos = self.last_state
            
            # --- Coordinate System Transformation ---
            base_x = center_x + pos * POS_SCALE
            x1 = base_x - L1 * math.sin(theta1)
            y1 = base_y + L1 * math.cos(theta1)
            x2 = x1 - L2 * math.sin(theta1 + theta2)
            y2 = y1 + L2 * math.cos(theta1 + theta2)

            # Validate coordinates before drawing
            if not all(math.isfinite(x) for x in [x1, y1, x2, y2, base_x]):
                self.draw_text("Invalid state data!", 300, 280, (255, 0, 0))
            else:
                # Draw pendulum
                pygame.draw.line(self.screen, (200, 200, 200), (base_x, base_y), (x1, y1), 6)
                pygame.draw.circle(self.screen, (180, 0, 0), (int(x1), int(y1)), 8)
                pygame.draw.line(self.screen, (100, 200, 100), (x1, y1), (x2, y2), 6)
                pygame.draw.circle(self.screen, (0, 0, 200), (int(x2), int(y2)), 8)
                pygame.draw.circle(self.screen, (255, 255, 0), (int(base_x), int(base_y)), 6)

            # Draw text
            self.draw_text(f"θ1: {math.degrees(theta1):+.2f}°", 10, 10)
            self.draw_text(f"θ2: {math.degrees(theta2):+.2f}°", 10, 35)
            self.draw_text(f"ω1: {vel1:+.3f} rad/s", 10, 60)
            self.draw_text(f"ω2: {vel2:+.3f} rad/s", 10, 85)
            self.draw_text(f"Pos: {pos:+.1f} steps", 10, 110)
            self.draw_text("←/→: Move | Space: Stop | Q: Quit", 10, 560)

            pygame.display.flip()
            self.clock.tick(FPS)

        # Cleanup
        self.esp.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    vis = PendulumVisualizer(SERIAL_PORT, BAUD_RATE)
    vis.run()