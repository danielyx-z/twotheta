#pragma once
#include <AccelStepper.h>

#define EN_PIN   26
#define STEP_PIN 14
#define DIR_PIN  12

extern AccelStepper stepper;

void setupMotor();
void moveStepper(float action);
void runStepper();