#pragma once

#include <cmath>

#include "util/vector3.hpp"
#include "util/random.hpp"
#include "util/colour.h"

class Particle {
  public:
	const type mass;
	const type size;
	vector3 position = {};
	vector3 velocity = {0, 0, 0};
	vector3 acceleration = {0, 0, 0};
	colourRGB colour = {255, 255, 255};

  public:
	// Particle(type mass, type size);
	// Particle(type mass, type size, vector3 position);
	// Particle(type mass, type size, vector3 position, vector3 velocity);
	// Particle(type mass, type size, vector3 position, vector3 velocity, vector3 acceleration);
	Particle(type mass = 1, type size = 5.0, vector3 position = /*randPer3()*/{500, 500, 0}, vector3 velocity = {0, 0, 0}, vector3 acceleration = {0, 0, 0}, colourRGB colour = {255, 255, 255});

	~Particle();

	vector3 &updatePos(const type dt);
	// vector3 &updateVel(const type dt);
	// vector3 &updateAcc();
	void applyForce(const vector3& forcePos,const int8_t force);

	void printParticle() const;
};