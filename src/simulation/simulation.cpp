
#include "simulation/simulation.hpp"

    // A lot of the implemantation can be found in the header file, because of the template and inline functions.

    Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
    Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps), window(Frame(frameWidth, frameHeight)), draw(Draw(window)), particles(std::vector<Particle>(particleAmount, Particle(5.0))) {}
    
    Simulation::~Simulation() {
        particles.clear();
        particles.shrink_to_fit();
    }

    