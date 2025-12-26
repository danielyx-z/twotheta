#include "Encoders.h"
#include <Wire.h>
#include <AS5600.h>

#define TCA9548A_ADDR 0x70
#define I2C_SDA_PIN 21
#define I2C_SCL_PIN 22
#define ENCODER1_CHANNEL 1 
#define ENCODER2_CHANNEL 0 

AS5600 encoder1;
AS5600 encoder2;

static uint16_t offset1 = 0;
static uint16_t offset2 = 0;

void tcaSelect(uint8_t channel) {
  Wire.beginTransmission(TCA9548A_ADDR);
  Wire.write(1 << channel);
  Wire.endTransmission();
}

void setupEncoders() {
  static bool enc1 = false;
  static bool enc2 = false;
  static unsigned long lastTry = 0;

  Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN);
  Wire.setClock(400000);

  while (!enc1 || !enc2) {
    if (millis() - lastTry >= 1000) {
      lastTry = millis();

      tcaSelect(ENCODER1_CHANNEL);
      if (!enc1 && encoder1.begin()) {
        offset1 = encoder1.rawAngle();
        enc1 = true;
        Serial.println("Encoder 1 ready");
      }

      tcaSelect(ENCODER2_CHANNEL);
      if (!enc2 && encoder2.begin()) {
        offset2 = encoder2.rawAngle();
        enc2 = true;
        Serial.println("Encoder 2 ready");
      }
    }
    delay(10);
  }
}

void getAngleAndVelocity(int joint, float &angle, float &velocity) {
  uint16_t raw = 0;

  if (joint == 1) {
    tcaSelect(ENCODER1_CHANNEL);
    raw = encoder1.rawAngle();
    raw = (raw - offset1) & 4095;
    // readAngle updates internal state for velocity calculation
    encoder1.readAngle(); 
    velocity = encoder1.getAngularSpeed(AS5600_MODE_RADIANS, false);
  } else if (joint == 2) {
    tcaSelect(ENCODER2_CHANNEL);
    raw = encoder2.rawAngle();
    raw = (raw - offset2) & 4095;
    encoder2.readAngle();
    velocity = encoder2.getAngularSpeed(AS5600_MODE_RADIANS, false);
  }

  angle = raw * AS5600_RAW_TO_RADIANS;
}