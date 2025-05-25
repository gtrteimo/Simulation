#include "simulation/particle.hpp"

// Particle::Particle(type mass = 1, type size = 1) : Particle(mass, size, randPer3(), randPer3(), randPer3()) {}
// Particle::Particle(type mass = 1, type size, vector3 position) : Particle(mass, size, position, randPer3(), randPer3()) {}
// Particle::Particle(type mass = 1, type size, vector3 position, vector3 velocity) : Particle(mass, size, position, velocity, randPer3()) {}
// Particle::Particle(type mass = 1, type size, vector3 position, vector3 velocity, vector3 acceleration) : Particle(mass, size, position, velocity, acceleration, {255, 255, 255}) {}

Particle::Particle(type mass, type size, vector3 position, vector3 velocity, vector3 acceleration, colourRGB colour) : mass(mass), size(size), position(position), velocity(velocity), acceleration(acceleration), colour(colour) {}

Particle::~Particle() {}

vector3& Particle::updatePos(type dt) {
    return position += velocity*dt;
}
vector3& Particle::updateVel(type dt) {
    return velocity += acceleration*dt;
}
vector3& Particle::updateAcc() {
    return acceleration = force * (1.0f / mass);
}
vector3& Particle::applyForce(vector3& f) {
    return force += f;
}
vector3& Particle::resetForce() {
    return force = {0, 0, 0};
}