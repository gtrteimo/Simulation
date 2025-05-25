#pragma once

#include <vector>

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

    template <typename... Fns> //Don't question this, just accept it as it is!!
    int loop(Fns&&... fns) { //Don't question this, just accept it as it is!!
        while (!glfwWindowShouldClose(window.getWindow())) {

            // std::cout << "FPS: " << fps << std::endl;

            updatePos();
            updateVel();
            updateAcc();
            std::vector<vector3> input;
            window.input(input);

            if (input[0].isValid()) {
                std::cout << "Mouse Left position: (" << input[0].x << ", " << input[0].y << ")" << std::endl;
                particles.clear();
            }
            if (input[1].isValid()) {
                particles.push_back(Particle(1.0, 0.1, input[1]));
                std::cout << "Mouse Right position: (" << input[1].x << ", " << input[1].y << ")" << std::endl;
            }
            


            (..., std::forward<Fns>(fns)()); //Don't question this, just accept it as it is!!

            // auto start = std::chrono::steady_clock::now();
            // while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(750)) {
                glfwPollEvents();
            //     if (glfwWindowShouldClose(window.getWindow())) {
            //         break; // Exit early if user tries to close the window
            //     }
            //     std::this_thread::sleep_for(std::chrono::milliseconds(10)); // Light CPU use
            // }
            draw.clearScreen({0, 0, 0});
            // draw.drawCircle({0.5, 0.5, 0}, 0.5, {255, 155, 0}, true, 36);
            draw.drawParticles(particles);
            draw.swapBuffers();

            // window.update();
        }
        return 0;
    }
};