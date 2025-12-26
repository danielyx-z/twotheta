#include "Motor.h"
#include <Arduino.h>

const int MAX_SPEED_HZ = 100000; 
const int ACCELERATION = 850000; 
const int ENDPOINT = 32000; 
const unsigned long HOLD_DURATION_MS = 500; 
const int LIMIT_SWITCH_PIN = 19; 
const int CENTER_OFFSET = 42500;

FastAccelStepperEngine engine;
FastAccelStepper *stepper = NULL;

unsigned long lastMoveTime = 0;
bool isEnabled = false;

void setupMotor() {
    engine.init();
    pinMode(LIMIT_SWITCH_PIN, INPUT_PULLUP);
    
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

void resetMotorPosition() {
    if (!stepper) return;

    stepper->enableOutputs();
    isEnabled = true;
    lastMoveTime = millis();

    stepper->setSpeedInHz(16000); 
    stepper->setAcceleration(20000);
    stepper->runForward();

    while (digitalRead(LIMIT_SWITCH_PIN) == HIGH) {
        delay(1); 
    }

    stepper->forceStop();
    delay(100); 
    
    stepper->setCurrentPosition(0);
    stepper->moveTo(-CENTER_OFFSET);

    while (stepper->isRunning()) {
        delay(1);
    }

    stepper->setCurrentPosition(0);
    stepper->setAcceleration(ACCELERATION);
}

void moveStepper(float action) {
    if (!stepper) return;

    action = constrain(action, -1.0f, 1.0f);
    long pos = stepper->getCurrentPosition();

    if ((pos >= ENDPOINT && action > 0) || (pos <= -ENDPOINT && action < 0)) {
        action = 0; 
    }

    uint32_t speed = (uint32_t)(abs(action) * MAX_SPEED_HZ);
    
    if (speed == 0) {
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

    if ((pos >= ENDPOINT && speed > 0) || (pos <= -ENDPOINT && speed < 0)) {
        stepper->forceStopAndNewPosition(pos); 
    }

    if (isEnabled && !stepper->isRunning()) {
        if (millis() - lastMoveTime >= HOLD_DURATION_MS) {
            stepper->disableOutputs();
            isEnabled = false;
        }
    }
}