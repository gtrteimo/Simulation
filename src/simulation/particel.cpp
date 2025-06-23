#include "simulation/particle.hpp" // Include the Particle class header

#include "util/log.hpp"


/**
 * Constructor...There are default values saved in the header since there are a lot of parameters here.
 * Mass can't be >= 0 as there is no such thing as negative mass and we also don't want divide by 0! 
 */
Particle::Particle(type mass, type size, vector2 position, vector2 velocity, colourRGB colour) : mass(mass), size(size), position(position), velocity(velocity), colour(colour){
    if (mass <= 0) {
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
    if (mass <= 0) {
        return; 
    } 
    
    if (!force.isEmpty()) {
        force -= position; // Currently all Particle have a force that is exactly -position when no other force is applied by clicking!
    } else {
        force = {0, 0};
    }

    acceleration = force / mass;
    force.empty();
    velocity += acceleration * dt;
}
void Particle::updatePos(type dt) {
    position += velocity * dt;
}

void Particle::applyForce(const vector2& newForce) {
    force.x += newForce.x; 
    force.y += newForce.y; 
}

bool Particle::wallCollision(type dt) {
    bool ret = false;
    vector2 checker = position + velocity * dt;
    if ((checker.x + size) >= 1 || (checker.x - size) <= -1) {
        velocity.x *= -FRICTION_MULTI;
        ret = true;
    }
    if ((checker.y + size) >= 1 || (checker.y - size) <= -1) {
        velocity.y *= -FRICTION_MULTI;
        ret = true;
    }
    return ret;
}

bool Particle::particelCollision(type dt, std::vector<Particle>& particles) {
    for (Particle& particle : particles) {
        vector2 intersection;
        if ((intersection = intersect(dt, particle)).isValid()) {
            std::cout<<"Hello Collision!\n"; 
        }
    }
    return false;
}   

/**
 * Sorry, a lot math and quadratic formula incoming!!
 * 
 * When 2 Circles collide the sum of there radii are exactly the distance between the two centers. Big revelation!
 * Thats how we get this: ||position1(x) - position2(x)|| = radius1 + radius2
 * 
 * To make our lives easier instead of having 2 moving circles we just have 1 stationary point at (0/0) and 
 *                       a Circle with the sum of the radii as radius, the relative position of the two as it's postion and lastly the relative velocity of the two as it's own velocity.
 * Here is the prosess: relativePosition(x) = position1(x) - position2(x) = (position1(0) + velocity1 * x) - (position2(0) + velocity2 * x)
 *                       = (position1(0) - position2(0)) + (velocity1(0) - velocity2(0) * x) = relativePosition(0) + relativeVelocity * x
 * 
 * Because of this conclusion we can say this instead: ||relativePosition(0) + relativeVelocity * x|| = radius1 + radius2
 * In our case || vector2 || is just sqrt(x^2 + y^2) as we are only dealing in 2 dimensions so we still have an annoying squareroot we want to get rid of
 * We get rid of that by simply squaring both sides: ||relativePosition(0) + relativeVelocity * x|| ^ 2 = (radius1 + radius2) ^ 2
 * 
 * But do you know what ||relativePosition(0) + relativeVelocity * x|| ^ 2 is the same as the dot product of relativePosition(0) + relativeVelocity * x
 * So the new formula look like this: (relativePosition(0) + relativeVelocity * x) * (relativePosition(0) + relativeVelocity * x) = (radius1 + radius2) ^ 2
 * 
 * We can unfold this even more so i will just simplify it : A = relativePosition(0), B = relativeVelocity * x, R = radius1+radius2
 * 
 * Currently the equation look like this: (A+B)^(A+B) = R^2 but we can use binomial expansion (A+B)^A * (A+B)^B now we can approximate it like this: A^A + B^B + A^B + B^A = R^2 //Note this is only a approximation and it is not the same thing
 * 
 * Without A, B and C it would look like this: (relativePosition(0) ** relativePosition(0)) + (relativePosition(0) ** relativeVelocity * t) + ((relativeVelocity * t) ** relativePosition(0)) + ((relativeVelocity * t) ** (relativeVelocity * t)) = (radius1 + radius2) ^2
 * 
 * 
 * 
 * WORKS!!! DO NOT TOUCH!!
 */
vector2 Particle::intersect(type dt, const Particle& particle) {
    vector2 ret = {-2, -2}; 

    vector2 relativePosition = position - particle.position;

    vector2 relativeVelocity = velocity - particle.velocity;

    type combinedRadius = size + particle.size;

    // Math: a*x^2 + b*x + c = 0
    type a = relativeVelocity.dot(relativeVelocity);
    type b = 2.0f * (relativePosition.dot(relativeVelocity));
    type c = relativePosition.dot(relativePosition) - (combinedRadius * combinedRadius);

    // Solve for t using the quadratic formula
    type discriminant = b * b - static_cast<type>(4.0) * a * c;
    if (discriminant >= 0) {
        type sqrtDiscriminant = std::sqrt(discriminant);
        type x1 = (-b - sqrtDiscriminant) / (static_cast<type>(2.0) * a);
        type x2 = (-b + sqrtDiscriminant) / (static_cast<type>(2.0) * a);

        type xImpact = static_cast<type>(-1.0); // Initialize to an invalid time

        // Find the earliest positive root within the dt interval
        if (x1 >= static_cast<type>(0.0) && x1 <= dt) {
            xImpact = x1;
        }
        if (x2 >= static_cast<type>(0.0) && x2 <= dt) {
            if (xImpact < static_cast<type>(0.0) || x2 < xImpact) { // check if it smaller then x1
                xImpact = x2;
            }
        }
        
        /*
         * If A is very small (velocities are almost parallel or zero), there are two possibilities 
         * either it's just a linear equation (since only a is effetely 0 and b is bigger) or
         * when both are close to zero 
         */
        if (std::abs(a) < static_cast<type>(0.00001)) { // If a is effectively zero, it's a linear equation or no movement
            if (std::abs(b) > static_cast<type>(0.00001)) { // Linear equation
                type xLinear = -c / b;
                if (xLinear >= static_cast<type>(0.0) && xLinear <= dt) {
                    xImpact = xLinear;
                }
            } else { // No relative velocity
                if (c <= static_cast<type>(0.0)) { // Already overlapping or touching
                    xImpact = static_cast<type>(0.0); // Collision at x=0
                }
            }
        }


        if (xImpact >= static_cast<type>(0.0) && xImpact <= dt) {
            // Collision occurred within the time step!
            // Calculate the positions of the centers at the time of impact
            vector2 impactPosition1 = position + velocity * xImpact;
            vector2 impactPosition2 = particle.position + particle.velocity * xImpact;

            // Calculate the vector from center1 to center2 at impact
            vector2 center_to_center_vec = impactPosition2 - impactPosition1;
            
            // Normalize this vector
            float impactDistance = center_to_center_vec.length();
            if (impactDistance > static_cast<type>(0.0)) { // Avoid division by zero if centers are coincident
                vector2 normal_direction = center_to_center_vec.normalize();

                // The direct collision point is on the surface of particle1,
                // in the direction of particle2, at its radius distance.
                ret = impactPosition1 + normal_direction * size;
            } else {
                // If centers are exactly coincident at impact, it's a special case (e.g., full overlap).
                // You might return one of the center positions or handle this differently based on your game logic.
                // For now, let's just return the center of the first particle.
                ret = impactPosition1;
            }
        }
    }

    return ret;
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