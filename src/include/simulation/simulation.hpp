#pragma once

#include <vector>
#include <functional>
#include <chrono>
#include <thread>

#include "util/vector2.hpp"
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
    inline  int updatePos();
  public:
    Simulation(uint16_t fps, uint64_t particleAmount);
	  Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount);
	  ~Simulation();

    int loop(std::function<int(std::vector<Particle>)> function);

    void addForce(const vector2& pos);
    void removeForce(const vector2& pos);
    void addParticle(const vector2& pos);
};