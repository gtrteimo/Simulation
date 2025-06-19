#include "simulation/simulation.hpp"

// This function initializes the simulation with a frame rate of 60 FPS and 5 particles,
// it can be further customised to include more parameters such as frame size or particle properties.
// It it just a simple Example to demonstrate how to start the simulation.

int initialiseGLFW () {
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
    return 0;
}

int print(std::vector<Particle> particles) {
    std::cout << particles.size() << "\n";
    // for (Particle particle: particles) {
    //     particle.printParticle();
    // }
    std::cout<<std::endl;
    return 0;
}

int startSimulation() {

    if (int ret = (initialiseGLFW()) < 0) {
        return ret;
    }

    Simulation test = Simulation(60, 0);
    test.loop(print);
    return 0;
}