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
// vector3& Particle::updateVel(type dt) {
//     return velocity += acceleration*dt;
// }
// vector3& Particle::updateAcc() {
//     return acceleration = force * (1.0f / mass);
// }
void Particle::applyForce(const vector3& forcePos, const int8_t force) {
    velocity.x += (force/(std::abs(forcePos.x - position.x) * std::abs(forcePos.x - position.x)));
    velocity.y += (force/(std::abs(forcePos.y - position.y) * std::abs(forcePos.y - position.y)));
    velocity.z += (force/(std::abs(forcePos.y - position.y) * std::abs(forcePos.y - position.y)));
}

void Particle::printParticle() const {
    std::cout << "Particle: { "
        << "Mass: " << mass << ", "
        << "Size: " << size << ", "
        << "Position: (" << position.x << "/" << position.y << "/" << position.z << "), "
        << "Velocity: (" << velocity.x << "/" << velocity.y << "/" << velocity.z << "), "
        // << "Acceleration: (" << acceleration.x << "/" << acceleration.y << "/" << acceleration.z << "), "
        // << "Force: (" << force.x << "/" << force.y << "/" << force.z << "), "
        << "Colour: (" << colour.r << "/" << colour.g << "/" << colour.b << ") "
        << "}";
}