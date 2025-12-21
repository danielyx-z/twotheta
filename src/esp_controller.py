import serial
import struct
import time
import math
import sys

class ESP32SerialController:
    STATE_HEADER = b'\xAA\x55'
    CMD_HEADER = b'\x55\xAA'
    STATE_PACKET_SIZE = 22
    CMD_PACKET_SIZE = 6

    def __init__(self, port, baudrate):
        self.serial = None
        try:
            self.serial = serial.Serial(port, baudrate, timeout=0)
            time.sleep(0.5)  # Allow ESP32 to reset
            self.serial.reset_input_buffer()
            self.serial.reset_output_buffer()
        except serial.SerialException as e:
            print(f"Error: Could not open serial port {port}: {e}")
            sys.exit(1)

    def move(self, action):
        """Sends a float action command to the ESP32."""
        if self.serial and self.serial.is_open:
            packet = self.CMD_HEADER + struct.pack('f', action)
            self.serial.write(packet)

    def receive_state(self):
        """Parses the latest state from the serial buffer."""
        if not (self.serial and self.serial.is_open):
            return None
        
        if self.serial.in_waiting < self.STATE_PACKET_SIZE:
            return None
        
        max_search = 200  
        search_count = 0
        
        while self.serial.in_waiting >= self.STATE_PACKET_SIZE and search_count < max_search:
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
                                # Skip ahead to the newest packet in the buffer
                                while self.serial.in_waiting >= self.STATE_PACKET_SIZE:
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