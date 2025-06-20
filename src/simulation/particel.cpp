#include "simulation/particle.hpp" // Include the Particle class header

Particle::Particle(type mass, type size, vector2 position, vector2 velocity, colourRGB colour) : mass(mass), size(size), position(position), velocity(velocity), acceleration({0, 0}), force({0, 0}), colour(colour){
    if (this->mass <= 0) {
        std::cerr << "Warning: Particle created with non-positive mass (" << this->mass << "). Setting mass to 1.0 to prevent division by zero.\n";
        const_cast<type&>(this->mass) = 1.0; 
    }
}

Particle::~Particle() {}

void Particle::updateAccVel(type dt) {
    if (mass <= 0) {
        return; 
    } 
    acceleration = force / mass;
        printParticle();
    force.empty();
    velocity += acceleration * dt;
    std::cout << "Acceleration: x: " << acceleration.x << ", y: " << acceleration.y << std::endl;
}
void Particle::updatePos(type dt) {
    position += velocity * dt;
}

void Particle::applyForce(const vector2& newForce) {
    force += newForce;
    std::cout << "Force: x: " << force.x << ", y: " << force.y << std::endl;
}

void Particle::printParticle() const {
    std::cout << "Particle: { "
        << "  Mass: " << mass << ", "
        << "  Size: " << size << ", "
        << "  Position: (" << position.x << ", " << position.y << "), "
        << "  Velocity: (" << velocity.x << ", " << velocity.y << "), "
        << "  Acceleration: (" << acceleration.x << ", " << acceleration.y << "), "
        << "  Force (pre-reset): (" << force.x << ", " << force.y << "), "
        << "  Colour: (R:" << static_cast<uint16_t>(colour.r) << ", G:" << static_cast<uint16_t>(colour.g) << ", B:" << static_cast<uint16_t>(colour.b) << ") "
        << "}\n";
}