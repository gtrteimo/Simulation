#include "simulation/particle.hpp" // Include the Particle class header

#include "util/log.hpp"

Particle::Particle(type mass, type size, vector2 position, vector2 velocity, colourRGB colour) : mass(mass), size(size), position(position), velocity(velocity), acceleration({0, 0}), force({0, 0}), colour(colour){
    if (this->mass <= 0) {
        std::cerr << "Warning: Particle created with non-positive mass (" << this->mass << "). Setting mass to 1.0 to prevent division by zero.\n";
        const_cast<type&>(this->mass) = 1.0; 
    }
}

Particle::~Particle() {}

vector2 Particle::getPosition() const {
    return position;
}
type Particle::getSize() const {
    return size;
}
colourRGB Particle::getColour() const {
    return colour;
}

void Particle::updateAccVel(type dt) {
        // std::cout << "Force: " << force <<  std::endl;
    if (mass <= 0) {
        return; 
    } 
    
    force -= position; // Currently all Particle have a force that is exactly -position when no other force is applied by clicking!

    acceleration = force / mass;
    // printParticle();
    force.empty();
    velocity += acceleration * dt;
    // std::cout << "Acceleration: " << acceleration << std::endl;
}
void Particle::updatePos(type dt) {
    position += velocity * dt;
}

void Particle::applyForce(const vector2& newForce) {
    force.x += newForce.x; 
    force.y += newForce.y; 

//     printForce();
//     std::cout << "Force: " << force <<  std::endl;
}

void Particle::printParticle() const {
    std::cout << "Particle: { "
        << "  Mass: " << mass << ", "
        << "  Size: " << size << ", "
        << "  Position: " << position << ", "
        << "  Velocity: " << velocity << ", "
        << "  Acceleration: " << acceleration << ", "
        << "  Force (pre-reset): " << force << ", "
        << "  Colour: (R:" << static_cast<uint16_t>(colour.r) << ", G:" << static_cast<uint16_t>(colour.g) << ", B:" << static_cast<uint16_t>(colour.b) << ") "
        << "}\n";
}