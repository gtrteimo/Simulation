#include "simulation/simulation.hpp"

int initialiseGLFW() {
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1; // Initialization failed
	}

	// Set global GLFW window hints (must be done before creating any window)
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__ // This is as far i'll go to support MacOS Users
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	return 0;
}

/*
 * These function initialize the simulation with a maximum frame rate of 144 FPS and 0 particles,
 * it can be further customized to do other things with the particles
 * Her i have included a function that prints all Particle details to the console.
 * Just have a function with the same parameters and same return value as the example print!
 * Note: All negative return values of the function are treated as errors and stop the loop
 * It it just a simple Example to demonstrate how to start the simulation.
 */

int print(std::vector<Particle> particles) {
	std::cout << "Particle Count: " << particles.size() << "\n";
	// for (Particle particle: particles) {
	//     particle.printParticle(); // Really slow
	// }
	std::cout << std::endl;
	return 0;
}

int startSimulation() {

	// Necessary don't remove
	if (int ret = (initialiseGLFW()) < 0) {
		return ret;
	}

	Simulation test = Simulation(144, 1);
	test.loop(print);
	return 0;
}