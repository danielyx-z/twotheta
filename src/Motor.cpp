#include "Motor.h"

const float MAX_SPEED = 20000.0f;
const int ENDPOINT = 10000;

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

static float commanded_action = 0.0f;
static float current_speed = 0.0f;

void setupMotor() {
  pinMode(EN_PIN, OUTPUT);
  digitalWrite(EN_PIN, LOW);

  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setSpeed(0.0f);
  stepper.setCurrentPosition(0);
}

void moveStepper(float action) {
  if (action > 1.0f) action = 1.0f;
  if (action < -1.0f) action = -1.0f;

  commanded_action = action;
}

void runStepper() {
  long pos = stepper.currentPosition();

  if ((pos >= ENDPOINT && commanded_action > 0.0f) ||
      (pos <= -ENDPOINT && commanded_action < 0.0f)) {
    current_speed = 0.0f;
    stepper.setSpeed(0.0f);
    return;
  }

  current_speed = commanded_action * MAX_SPEED;
  stepper.setSpeed(current_speed);

  stepper.runSpeed();
}
