#include "simulation/simulation.hpp"

int startSimulation() {
    Simulation test = Simulation(60, 5);
    test.loop();
    return 0;
}