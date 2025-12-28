#include "Motor.h"
#include <Arduino.h>

const int MAX_SPEED_HZ = 100000; 
const int ACCELERATION = 800000; 
const int ENDPOINT = 32000; // Reduced slightly for safety margin
const int SAFETY_ZONE = 200; // Steps before endpoint to start slowing down
const unsigned long HOLD_DURATION_MS = 1000; 
const int LIMIT_SWITCH_PIN = 19; 
const int CENTER_OFFSET = 45000;

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
    if (stepper->isRunning()) {
        stepper->setSpeedInHz(0); 
        // Vital: Wait for the library to confirm it has truly stopped
        while (stepper->isRunning()) {
            delay(1);
        }
        delay(50); // Small settle time
    }
    delay(50); // Short pause to let physical vibration settle
    stepper->enableOutputs();
    isEnabled = true;
    lastMoveTime = millis();

    stepper->setSpeedInHz(12000); 
    stepper->setAcceleration(20000);
    stepper->runForward();

    while (digitalRead(LIMIT_SWITCH_PIN) == HIGH) {
        delay(1); 
    }
    stepper->setSpeedInHz(20000); 
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

    if (pos >= ENDPOINT && action > 0) {
        stepper->forceStopAndNewPosition(ENDPOINT);
        return;
    }
    if (pos <= -ENDPOINT && action < 0) {
        stepper->forceStopAndNewPosition(-ENDPOINT);
        return;
    }

    float speed_scale = 1.0f;
    if (action > 0) {
        long dist = ENDPOINT - pos;
        if (dist < SAFETY_ZONE) speed_scale = (float)dist / (float)SAFETY_ZONE;
    } else if (action < 0) {
        long dist = pos - (-ENDPOINT);
        if (dist < SAFETY_ZONE) speed_scale = (float)dist / (float)SAFETY_ZONE;
    }

    uint32_t target_speed = (uint32_t)(abs(action) * MAX_SPEED_HZ * speed_scale);
    
    if (target_speed < 100) {
        stepper->stopMove(); 
    } else {
        if (!isEnabled) {
            stepper->enableOutputs();
            isEnabled = true;
        }
        lastMoveTime = millis(); 
        stepper->setSpeedInHz(target_speed);
        
        if (action > 0) stepper->moveTo(ENDPOINT);
        else stepper->moveTo(-ENDPOINT);
    }
}

void checkMotorSafety() {
    if (!stepper) return;

    long pos = stepper->getCurrentPosition();

    if (pos > ENDPOINT) {
        stepper->forceStopAndNewPosition(ENDPOINT);
    } else if (pos < -ENDPOINT) {
        stepper->forceStopAndNewPosition(-ENDPOINT);
    }

    if (isEnabled && !stepper->isRunning()) {
        if (millis() - lastMoveTime >= HOLD_DURATION_MS) {
            stepper->disableOutputs();
            isEnabled = false;
        }
    }
}