#pragma once

#include <vector>

#include "util/vector3.hpp"
#include "util/colour.h"

#include "gui/frame.hpp"
#include "simulation/particle.hpp"

class Simulation {
  private:
    const uint16_t fps;
    Frame window;
    std::vector<Particle> particles;
  public:
    Simulation(uint16_t fps, uint64_t particleAmount);
	  Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount);
	  ~Simulation();

    int loop();
  private:
    inline int updatePos();
    inline int updateVel();
    inline int updateAcc();
};