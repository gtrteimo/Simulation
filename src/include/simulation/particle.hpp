#pragma once

#include "util/types.h"
#include "util/vector2.hpp"
#include "util/colour.h"
#include "util/random.hpp"

#include <iostream>

class Particle {
  public:
    const type mass;
    const type size;
    vector2 position;
    vector2 velocity;
    vector2 acceleration;
    vector2 force;
    colourRGB colour;

  public:
    Particle(type mass = 1.0, type size = 0.05, vector2 position = {0, 0}, vector2 velocity = {0, 0}, colourRGB colour = randLightColour());

    ~Particle();

	void updateAccVel(type dt);
    void updatePos(type dt); 
    
    void applyForce(const vector2& force);

    void printParticle() const;
};