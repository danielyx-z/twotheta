#pragma once
#include <FastAccelStepper.h>

// Pin Definitions
#define EN_PIN   26
#define STEP_PIN 14
#define DIR_PIN  12

// Expose the stepper object if needed elsewhere (optional)
extern FastAccelStepper *stepper;

void setupMotor();
void moveStepper(float action);
void checkMotorSafety(); // Call this in loop to enforce endpoints
void resetMotorPosition(); 