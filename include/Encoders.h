#ifndef ENCODERS_H
#define ENCODERS_H

#include <Arduino.h>

void setupEncoders();
float getAngle(int joint);
float getAngularVelocity(int joint);

#endif // ENCODERS_H
