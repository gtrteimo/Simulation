#pragma once

#include "util/types.h"
#include "util/vector2.hpp"
#include "util/colour.h"
#include "util/random.hpp"

#include <iostream>

const static type FRICTION_MULTI = 0.80;

class Particle {
  private:
    const type mass;
    const type size;
    vector2 position;
    vector2 velocity;
    vector2 acceleration = {0, 0};
    vector2 force = {0, 0};
    colourRGB colour;

  public:
    Particle(type mass = 1.0, type size = 0.01, vector2 position = {0, 0}, vector2 velocity = {0, 0}, colourRGB colour = randSingleColour());

    ~Particle();

    vector2 getPosition() const;
    type getSize() const;
    colourRGB getColour() const;

	  void updateAccVel(type dt);
    void updatePos(type dt); 
    
    void applyForce(const vector2& force);

    void printParticle() const;

    bool wallCollision(type dt);
    bool particelCollision(type dt, std::vector<Particle>& particles);
  private:
    vector2 intersect(type dt, const Particle& particel);
};