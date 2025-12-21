#include "Motor.h"

const int MAX_SPEED_HZ = 24000; 
const int ACCELERATION = 600000; 
const int ENDPOINT = 12000;      

FastAccelStepperEngine engine;
FastAccelStepper *stepper = NULL;

void setupMotor() {
  engine.init();
  stepper = engine.stepperConnectToPin(STEP_PIN);
  if (stepper) {
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(EN_PIN);
    stepper->setAutoEnable(true); 
    stepper->setAcceleration(ACCELERATION);
  }
}

void moveStepper(float action) {
  if (!stepper) return;

  action = constrain(action, -1.0f, 1.0f);
  long pos = stepper->getCurrentPosition();

  // FIX: Only block if the action moves further into the forbidden zone
  // Allow the action if it moves back toward zero
  if (pos >= ENDPOINT && action > 0) {
    stepper->stopMove();
    return;
  }
  if (pos <= -ENDPOINT && action < 0) {
    stepper->stopMove();
    return;
  }

  uint32_t speed = abs(action) * MAX_SPEED_HZ;
  
  if (speed == 0) {
    stepper->stopMove();
  } else {
    stepper->setSpeedInHz(speed);
    if (action > 0) stepper->runForward();
    else stepper->runBackward();
  }
}

void checkMotorSafety() {
  if (!stepper) return;

  long pos = stepper->getCurrentPosition();
  
  // Get speed in milliHz. Positive means moving forward, negative backward.
  int32_t speed = stepper->getCurrentSpeedInMilliHz();

  // If past positive limit and still moving forward
  if (pos >= ENDPOINT && speed > 0) {
      stepper->forceStopAndNewPosition(pos); 
  }
  // If past negative limit and still moving backward
  else if (pos <= -ENDPOINT && speed < 0) {
      stepper->forceStopAndNewPosition(pos);
  }
}