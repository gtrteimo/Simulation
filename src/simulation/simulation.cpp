
#include "simulation/simulation.hpp"

    Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
    Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps+1), window(Frame(frameWidth, frameHeight)), draw(Draw(window)), particles(std::vector<Particle>(particleAmount, Particle(5.0))) {}
    
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

        const std::chrono::nanoseconds frameTime(static_cast<long long>(1e9 / fps)); // Duration of one frame in nanoseconds

        while (!glfwWindowShouldClose(window.getWindow())) {

            //Timer start
            auto start = std::chrono::high_resolution_clock::now();

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

            auto end = std::chrono::high_resolution_clock::now();

            auto duration = end - start;

            auto target_end_time = start + frameTime;
            
             if (duration < frameTime) {
                std::this_thread::sleep_until(target_end_time);
            }

            auto end2 = std::chrono::high_resolution_clock::now();

            auto duration2 = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - start);

            double currentFps = static_cast<double>(1e9) / static_cast<double>(duration2.count());
            std::cout << "FPS: " << static_cast<int>(currentFps) << std::endl;
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

    