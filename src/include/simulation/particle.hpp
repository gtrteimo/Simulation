#pragma once

#include <cmath>

#include "util/vector2.hpp"
#include "util/random.hpp"
#include "util/colour.h"

class Particle {
  public:
	const type mass;
	const type size;
	vector2 position = {};
	vector2 velocity = {0, 0};
	colourRGB colour = {255, 255, 255};

  public:
	// Particle(type mass, type size);
	// Particle(type mass, type size, vector3 position);
	// Particle(type mass, type size, vector3 position, vector3 velocity);
	// Particle(type mass, type size, vector3 position, vector3 velocity, vector3 acceleration);
	Particle(type mass = 1, type size = 5.0, vector2 position = /*randPer2()*/{500, 500}, vector2 velocity = {0, 0}, colourRGB colour = {255, 255, 255});

	~Particle();

	vector2 &updatePos(const type dt);
	void applyForce(const vector2& forcePos,const int8_t force);

	void printParticle() const;
};