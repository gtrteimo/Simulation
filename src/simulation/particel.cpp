#include "simulation/particle.hpp"

Particle::Particle(type mass, type size, vector2 position, vector2 velocity, colourRGB colour) : mass(mass), size(size), position(position), velocity(velocity), colour(colour) {}

Particle::~Particle() {}

vector2& Particle::updatePos(type dt) {
    return position += velocity*dt;
}
// vector3& Particle::updateVel(type dt) {
//     return velocity += acceleration*dt;
// }
// vector3& Particle::updateAcc() {
//     return acceleration = force * (1.0f / mass);
// }
void Particle::applyForce(const vector2& forcePos, const int8_t force) {
    velocity.x += (force/(std::abs(forcePos.x - position.x) * std::abs(forcePos.x - position.x)));
    velocity.y += (force/(std::abs(forcePos.y - position.y) * std::abs(forcePos.y - position.y)));
}

void Particle::printParticle() const {
    std::cout << "Particle: { "
        << "Mass: " << mass << ", "
        << "Size: " << size << ", "
        << "Position: (" << position.x << "/" << position.y << "), "
        << "Velocity: (" << velocity.x << "/" << velocity.y << "), "
        << "Colour: (" << colour.r << "/" << colour.g << "/" << colour.b << ") "
        << "}";
}