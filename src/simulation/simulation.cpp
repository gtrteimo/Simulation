
#include "simulation/simulation.hpp"

    Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
    Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps), window(Frame(frameWidth, frameHeight)), draw(Draw(window)), particles(std::vector<Particle>(particleAmount, Particle(5.0))) {}
    
    Simulation::~Simulation() {
        particles.clear();
        particles.shrink_to_fit();
    }

    inline int Simulation::updatePos() {
        if (fps == 0) {
            return -1;
        }
        type dt = static_cast<type>(1.0) / static_cast<type>(fps);
        for (Particle& particle : particles) {
            particle.updatePos(dt);
        }
        return 0;
    }

    int Simulation::loop(std::function<int(std::vector<Particle>)> function) {
        while (!glfwWindowShouldClose(window.getWindow())) {

            //Timer start
            auto start_of_iteration = std::chrono::high_resolution_clock::now();

            // std::cout << "FPS: " << fps << std::endl;

            std::vector<vector3> input;
            window.input(input);

            //Left click
            if (input[0].isValid()) {
                // std::cout << "Mouse Left position: (" << input[0].x << ", " << input[0].y << ")" << std::endl;
                particles.clear();
            }
            //Right click
            if (input[1].isValid()) {
                // std::cout << "Mouse Right position: (" << input[1].x << ", " << input[1].y << ")" << std::endl;
                addParticle(input[1]);
            }
            
            if (int ret = (function(particles)) < 0) {
                return ret;
            }

            updatePos();
            
            draw.clearScreen({0, 0, 0});
            draw.drawParticles(particles);
            draw.swapBuffers();

            glfwPollEvents();

            auto end_of_iteration = std::chrono::high_resolution_clock::now();

            long long iteration_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end_of_iteration - start_of_iteration).count(); 

            std::cout << "Duration: " << iteration_duration_us << " microseconds" << std::endl;
        }
        return 0;
    }

    void Simulation::addForce(const vector3& pos) {
        for (Particle particle : particles) {
            particle.applyForce(pos, 5);
        }
    }
    void Simulation::removeForce(const vector3& pos) {
        for (Particle particle : particles) {
            particle.applyForce(pos, -5);
        }
    }
    void Simulation::addParticle(const vector3& pos) {
        particles.push_back(Particle(1.0, 0.1, pos));
    }

    