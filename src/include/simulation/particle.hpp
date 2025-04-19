#pragma once

#include "util/vector3.hpp"
#include "util/random.hpp"
#include "util/colour.h"

class Particle {
  private:
  	const type mass;
	vector3 position = {};
	vector3 velocity = {0, 0, 0};
	vector3 acceleration = {0, 0, 0};
	vector3 force = {0, 0, 0};
	colourRGB colour = {255, 255, 255};	
  public:
  	Particle(type mass);
  	Particle(type mass, vector3 position);
  	Particle(type mass, vector3 position, vector3 velocity);
	Particle(type mass, vector3 position, vector3 velocity, vector3 acceleration);
	Particle(type mass, vector3 position, vector3 velocity, vector3 acceleration, colourRGB colour);

	~Particle();

	vector3& updatePos(type dt);
	vector3& updateVel(type dt);
	vector3& updateAcc();
	vector3& applyForce(vector3& f);
	vector3& resetForce();
};