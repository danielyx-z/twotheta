#include "Motor.h"
#include <Arduino.h>

const int MAX_SPEED_HZ = 100000; 
const int ACCELERATION = 800000; 
const int ENDPOINT = 32000; 
const unsigned long HOLD_DURATION_MS = 500; 

FastAccelStepperEngine engine;
FastAccelStepper *stepper = NULL;

unsigned long lastMoveTime = 0;
bool isEnabled = false;

void setupMotor() {
  engine.init();
  stepper = engine.stepperConnectToPin(STEP_PIN);
  if (stepper) {
    stepper->setDirectionPin(DIR_PIN);
    stepper->setEnablePin(EN_PIN);
    stepper->setAutoEnable(false); 
    stepper->setAcceleration(ACCELERATION);
    stepper->disableOutputs();
    isEnabled = false;
  }
}

// --- NEW: Robust Reset Function ---
void resetMotorPosition() {
  if (!stepper) return;

  // 1. Ensure motor is enabled
  if (!isEnabled) {
    stepper->enableOutputs();
    isEnabled = true;
  }

  // 2. Update timer so safety doesn't kill it immediately
  lastMoveTime = millis();

  // 3. Configure for homing
  stepper->setSpeedInHz(4000);   // Slower, safe speed for resetting
  stepper->setAcceleration(8000); 
  
  // 4. Move to absolute center
  stepper->moveTo(0);
}

void moveStepper(float action) {
  if (!stepper) return;

  // Clip action
  action = constrain(action, -1.0f, 1.0f);
  long pos = stepper->getCurrentPosition();

  // Boundary Protection
  if ((pos >= ENDPOINT && action > 0) || (pos <= -ENDPOINT && action < 0)) {
    action = 0; 
  }

  uint32_t speed = (uint32_t)(abs(action) * MAX_SPEED_HZ);
  
  if (speed == 0) {
    // Only stop if we are NOT currently resetting (moveTo)
    // We detect "velocity mode" by checking if target position is not set/same as current
    // But simplistic check: Just stop. resetMotorPosition will override this if called.
    stepper->stopMove(); 
  } else {
    if (!isEnabled) {
      stepper->enableOutputs();
      isEnabled = true;
    }
    lastMoveTime = millis(); 
    stepper->setSpeedInHz(speed);
    
    if (action > 0) stepper->runForward();
    else stepper->runBackward();
  }
}

void checkMotorSafety() {
  if (!stepper) return;

  long pos = stepper->getCurrentPosition();
  int32_t speed = stepper->getCurrentSpeedInMilliHz();

  // 1. Hard Limit Stop
  // Note: speed is signed. If pos > endpoint and moving positive (away from 0), STOP.
  if ((pos >= ENDPOINT && speed > 0) || (pos <= -ENDPOINT && speed < 0)) {
      stepper->forceStopAndNewPosition(pos); 
  }

  // 2. Timeout / Disable Logic
  if (isEnabled && !stepper->isRunning()) {
    if (millis() - lastMoveTime >= HOLD_DURATION_MS) {
      stepper->disableOutputs();
      isEnabled = false;
    }
  }
}