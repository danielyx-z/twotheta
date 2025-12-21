#include <Arduino.h>
#include "Motor.h"
#include "Encoders.h" // Ensure this file exists in your project

// Protocol Config
const unsigned long SERIAL_INTERVAL = 1; // ms
const uint8_t STATE_HEADER[2] = {0xAA, 0x55};
const uint8_t CMD_HEADER[2] = {0x55, 0xAA};
const int CMD_PACKET_SIZE = 6;

// Globals
unsigned long lastSerialTime = 0;
uint8_t serialBuffer[CMD_PACKET_SIZE];
int serialBufferIndex = 0;

void setup() {
  Serial.begin(921600);
  delay(500);

  setupMotor();
  setupEncoders(); // Remove if you don't have this file
  
  Serial.println("Ok.");
}

void loop() {
  // 1. Process Serial Input
  while (Serial.available()) {
    uint8_t inByte = Serial.read();

    // Simple header detection state machine
    if (serialBufferIndex == 0) {
      if (inByte == CMD_HEADER[0]) serialBuffer[serialBufferIndex++] = inByte;
    } else if (serialBufferIndex == 1) {
      if (inByte == CMD_HEADER[1]) serialBuffer[serialBufferIndex++] = inByte;
      else serialBufferIndex = 0;
    } else {
      serialBuffer[serialBufferIndex++] = inByte;

      if (serialBufferIndex >= CMD_PACKET_SIZE) {
        float action;
        memcpy(&action, serialBuffer + 2, 4); // Extract float
        
        if (isfinite(action)) {
          moveStepper(action); // Updates hardware timer immediately
        }
        serialBufferIndex = 0;
      }
    }
  }

  // 2. Safety Limit Check (Crucial for FastAccelStepper)
  checkMotorSafety();

  // 3. Send Telemetry
  unsigned long now = millis();
  if (now - lastSerialTime >= SERIAL_INTERVAL) {
    lastSerialTime = now;

    // Gather data
    float t1 = getAngle(1);
    float t2 = getAngle(2);
    float v1 = getAngularVelocity(1);
    float v2 = getAngularVelocity(2);
    float p = (float)stepper->getCurrentPosition();

    // Pack buffer
    uint8_t buffer[22];
    buffer[0] = STATE_HEADER[0];
    buffer[1] = STATE_HEADER[1];
    memcpy(buffer + 2, &t1, 4);
    memcpy(buffer + 6, &t2, 4);
    memcpy(buffer + 10, &v1, 4);
    memcpy(buffer + 14, &v2, 4);
    memcpy(buffer + 18, &p, 4);

    // Non-blocking write
    if (Serial.availableForWrite() >= 22) {
      Serial.write(buffer, 22);
    }
  }
}