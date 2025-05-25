#include "simulation/simulation.hpp"

// This function initializes the simulation with a frame rate of 60 FPS and 5 particles,
// it can be further customised to include more parameters such as frame size or particle properties.
// It it just a simple Example to demonstrate how to start the simulation.

int print() {
    // std::cout << "Simulation running!" << std::endl;
    return 0;
}

int startSimulation() {

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1; // Initialization failed
    }

        // Set global GLFW window hints (must be done before creating any window)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    Simulation test = Simulation(60, 0);
    test.loop(print, print, print, print, print);
    return 0;
}