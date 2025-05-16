
#include "simulation/simulation.hpp"

    Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
    Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps), window(Frame(frameWidth, frameHeight)), particles(std::vector<Particle>(particleAmount, Particle(5.0))) {}
    
    Simulation::~Simulation() {
        particles.clear();
        particles.shrink_to_fit();
    }

    inline int Simulation::updatePos() { 
        for (Particle& particle : particles) {
            particle.updatePos(((type)1)/fps);
        }
        return 0;
    }
    inline int Simulation::updateVel() {
        for (Particle& particle : particles) {
            particle.updateVel(((type)1)/fps);
        }
        return 0;
    }
    inline int Simulation::updateAcc() {
        for (Particle& particle : particles) {
            particle.updateAcc();
        }
        return 0;
    }

    int Simulation::loop() {
        while (!glfwWindowShouldClose(window.getWindow())) {
            updatePos();
            updateVel();
            updateAcc();
            window.input();
            window.update();
        }
        return 0;
    }