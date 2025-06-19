
#include "simulation/simulation.hpp"

    // A lot of the implemantation can be found in the header file, because of the template and inline functions.

    Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
    Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps), window(Frame(frameWidth, frameHeight)), draw(Draw(window)), particles(std::vector<Particle>(particleAmount, Particle(5.0))) {}
    
    Simulation::~Simulation() {
        particles.clear();
        particles.shrink_to_fit();
    }

    int Simulation::loop(std::function<int(std::vector<Particle>)> function) {
        while (!glfwWindowShouldClose(window.getWindow())) {

            auto start_of_iteration = std::chrono::high_resolution_clock::now();

            // std::cout << "FPS: " << fps << std::endl;

            updatePos();
            updateVel();
            updateAcc();
            std::vector<vector3> input;
            window.input(input);

            if (input[0].isValid()) {
                // std::cout << "Mouse Left position: (" << input[0].x << ", " << input[0].y << ")" << std::endl;
                particles.clear();
            }
            if (input[1].isValid()) {
                particles.push_back(Particle(1.0, 0.1, input[1]));
                // std::cout << "Mouse Right position: (" << input[1].x << ", " << input[1].y << ")" << std::endl;
            }
            
            if (int ret = (function(particles)) < 0) {
                return ret;
            }

            glfwPollEvents();
            draw.clearScreen({0, 0, 0});
            draw.drawParticles(particles);
            draw.swapBuffers();

            auto end_of_iteration = std::chrono::high_resolution_clock::now();

            long long iteration_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_of_iteration - start_of_iteration).count(); 

            std::cout << "Duration: " << iteration_duration_us << " microseconds" << std::endl;
        }
        return 0;
    }

    