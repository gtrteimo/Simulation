#pragma once

#include "util/types.h"
#include "util/vector2.hpp"
#include "util/colour.h"
#include "util/random.hpp"

#include <iostream>

const static type FRICTION_MULTI = 0.80;

class Particle {
  public:
    type mass;
    type size;
    vector2 position;
    vector2 velocity;
    vector2 acceleration = {0, 0};
    vector2 force = {0, 0};
    colourRGB colour;

  public:
    Particle(type mass = 1.0, type size = 0.05, vector2 position = randPos2(), vector2 velocity = {0, 0}, colourRGB colour = randSingleColour());

    ~Particle();

    vector2 getPosition() const;
    type getSize() const;
    colourRGB getColour() const;

	  void updateAccVel(type dt);
    void updatePos(type dt); 
    
    void applyForce(const vector2& force);

    bool wallCollision(type dt);
    bool particleCollision(type dt, std::vector<Particle>& particles);
  private:
    vector2 intersect(type dt, const Particle& particel);
};

std::ostream& operator<<(std::ostream& os, const Particle& vec);