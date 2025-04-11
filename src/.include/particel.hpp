#pragma once

#include "vector3.h"

class Particel {
  private:
	vector3 position = {};
	vector3 velocity = {0, 0, 0};
	vector3 acceleration = {0, 0, 0};
	const type mass;

  public:
	Particel(type mass);
	~Particel();
};