#include <Arduino.h>
#include "Motor.h"
#include "Encoders.h"

\
const unsigned long serialStreamInterval = 1; 
// Protocol constants
const uint8_t STATE_HEADER[2] = {0xAA, 0x55};  // Header for state packets
const uint8_t CMD_HEADER[2] = {0x55, 0xAA};    // Header for command packets
const int CMD_PACKET_SIZE = 6;  // 2 header + 4 data bytes


unsigned long lastSerialStreamTime = 0;
uint8_t serialBuffer[CMD_PACKET_SIZE];
int serialBufferIndex = 0;

void setup() {
  Serial.begin(921600);
  delay(1000);
  setupMotor();
  setupEncoders();
  
  Serial.println("Ok.");
  Serial.flush();
}

void loop() {
  while (Serial.available()) {
    uint8_t inByte = Serial.read();
    
    if (serialBufferIndex == 0) {
      if (inByte == CMD_HEADER[0]) {
        serialBuffer[serialBufferIndex++] = inByte;
      }
    } else if (serialBufferIndex == 1) {
      if (inByte == CMD_HEADER[1]) {
        serialBuffer[serialBufferIndex++] = inByte;
      } else {
        serialBufferIndex = 0;
        if (inByte == CMD_HEADER[0]) {
          serialBuffer[serialBufferIndex++] = inByte;
        }
      }
    } else {
      serialBuffer[serialBufferIndex++] = inByte;
      
      if (serialBufferIndex >= CMD_PACKET_SIZE) {
        float action;
        memcpy(&action, serialBuffer + 2, 4);  // Skip 2-byte header
        
        if (isfinite(action)) {
          action = constrain(action, -1.0f, 1.0f);
          moveStepper(action);
        }
        
        serialBufferIndex = 0;
      }
    }
  }

  unsigned long now = millis();
  if (now - lastSerialStreamTime >= serialStreamInterval) {
    lastSerialStreamTime = now;

    float t1 = getAngle(1);
    float t2 = getAngle(2);
    float v1 = getAngularVelocity(1);
    float v2 = getAngularVelocity(2);
    float p = stepper.currentPosition();

    // 2-byte header + 5 floats (2 + 20 = 22 bytes)
    uint8_t buffer[22];
    buffer[0] = STATE_HEADER[0];  // 0xAA
    buffer[1] = STATE_HEADER[1];  // 0x55
    memcpy(buffer + 2, &t1, 4);
    memcpy(buffer + 6, &t2, 4);
    memcpy(buffer + 10, &v1, 4);
    memcpy(buffer + 14, &v2, 4);
    memcpy(buffer + 18, &p, 4);

    Serial.write(buffer, 22);
  }

  runStepper();
  
  yield();  // should this be changed to yield(); ?? for faster performance.
}