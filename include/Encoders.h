#ifndef ENCODERS_H
#define ENCODERS_H

#include <Arduino.h>

void setupEncoders();
void getAngleAndVelocity(int joint, float &angle, float &velocity);
void recalibrateEncoders();
#endif // ENCODERS_H
