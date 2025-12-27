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
            print("ESP32 connection initialized.")
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
        
        buffer_size = self.serial.in_waiting
        if buffer_size < self.STATE_PACKET_SIZE:
            return None
        
        # Read everything at once - ONE syscall instead of thousands
        all_data = self.serial.read(buffer_size)
        
        # Find the LAST valid packet
        state = None
        i = len(all_data) - self.STATE_PACKET_SIZE
        
        while i >= 0:
            if all_data[i] == 0xAA and all_data[i+1] == 0x55:
                try:
                    new_state = struct.unpack('5f', all_data[i+2:i+22])
                    if all(math.isfinite(x) for x in new_state):
                        return new_state
                except struct.error:
                    pass
            i -= 1
        
        return state
    def get_buffer_size(self):
        return self.serial.in_waiting if self.serial else 0

    def close(self):
        if self.serial and self.serial.is_open:
            self.serial.close()