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
  public:
    static Simulation* sim;
  private:
    inline  int updateAccVel();
    inline  int updatePos();
  public:
    Simulation(uint16_t fps, uint64_t particleAmount);
	  Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount);
	  ~Simulation();

    int loop(std::function<int(std::vector<Particle>)> function);

    int inputHold();
    static void inputMouseTap(GLFWwindow* window, int button, int action, int mods);
    static void inputKeyboardTap(GLFWwindow* window, int key, int scancode, int action, int mods);

    void addForce(const vector2& pos);
    void removeForce(const vector2& pos);
    void addParticle(const vector2& pos, uint64_t amount);
};