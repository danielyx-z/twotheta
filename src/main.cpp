#include <Arduino.h>
#include "Motor.h"
#include "Encoders.h" 

const unsigned long SERIAL_INTERVAL = 10; 
const uint8_t STATE_HEADER[2] = {0xAA, 0x55};
const uint8_t CMD_HEADER[2] = {0x55, 0xAA};
const int CMD_PACKET_SIZE = 6;

unsigned long lastSerialTime = 0;
uint8_t serialBuffer[CMD_PACKET_SIZE];
int serialBufferIndex = 0;

void setup() {
  Serial.begin(921600);
  delay(500);
  pinMode(25, OUTPUT);
  pinMode(24, OUTPUT);
  digitalWrite(25, HIGH);
  digitalWrite(24, HIGH);
  setupMotor();
  setupEncoders();
  
  Serial.println("Ok.");
}

void loop() {
  // 1. Process Serial
  while (Serial.available()) {
    uint8_t inByte = Serial.read();

    if (serialBufferIndex == 0) {
      if (inByte == CMD_HEADER[0]) serialBuffer[serialBufferIndex++] = inByte;
    } else if (serialBufferIndex == 1) {
      if (inByte == CMD_HEADER[1]) serialBuffer[serialBufferIndex++] = inByte;
      else serialBufferIndex = 0;
    } else {
      serialBuffer[serialBufferIndex++] = inByte;

      if (serialBufferIndex >= CMD_PACKET_SIZE) {
        float action;
        memcpy(&action, serialBuffer + 2, 4); 
        
        if (isfinite(action)) {
          if (action > 5.0) { // > 5 = home
             resetMotorPosition();
          } 
          else {
              moveStepper(action); 
          }
        }
        serialBufferIndex = 0;
      }
    }
  }

  // 2. Safety
  checkMotorSafety();

  // 3. Telemetry
  unsigned long now = millis();
  if (now - lastSerialTime >= SERIAL_INTERVAL) {
    lastSerialTime = now;

    float t1, v1, t2, v2;
    getAngleAndVelocity(1, t1, v1);
    getAngleAndVelocity(2, t2, v2);
    float p = (float)stepper->getCurrentPosition();


    uint8_t buffer[22];
    buffer[0] = STATE_HEADER[0];
    buffer[1] = STATE_HEADER[1];
    memcpy(buffer + 2, &t1, 4);
    memcpy(buffer + 6, &t2, 4);
    memcpy(buffer + 10, &v1, 4);
    memcpy(buffer + 14, &v2, 4);
    memcpy(buffer + 18, &p, 4);

    if (Serial.availableForWrite() >= 22) {
      Serial.write(buffer, 22);
    }
  }
}