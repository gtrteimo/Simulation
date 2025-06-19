#pragma once

#include <vector>
#include <functional>
#include <chrono>
#include <thread>

#include "util/vector3.hpp"
#include "util/colour.h"

#include "gui/frame.hpp"
#include "simulation/draw.hpp"
#include "simulation/particle.hpp"

class Simulation {
  private:
    const uint16_t fps;
    Frame window;
    Draw draw;
    std::vector<Particle> particles;

  private:
    inline  int updatePos() {
        if (fps == 0) {
            return -1;
        }
        type dt = static_cast<type>(1.0) / static_cast<type>(fps);
        for (Particle& particle : particles) {
            particle.updatePos(dt);
        }
        return 0;
    }
    inline int updateVel() {
        if (fps == 0) {
          return -1;
        }
        type dt = static_cast<type>(1.0) / static_cast<type>(fps);
        for (Particle& particle : particles) {
            particle.updateVel(dt);
        }
        return 0;
    }
    inline int updateAcc() {
        for (Particle& particle : particles) {
            particle.updateAcc();
        }
        return 0;
    }
  public:
    Simulation(uint16_t fps, uint64_t particleAmount);
	  Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount);
	  ~Simulation();

    int loop(std::function<int(std::vector<Particle>)> function);
};