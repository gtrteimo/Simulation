
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

        // Duration of one frame in nanoseconds
        const std::chrono::nanoseconds frameTime(static_cast<long long>(1e9 / fps)); 

        while (!glfwWindowShouldClose(window.getWindow())) {

            //Timer start
            auto start = std::chrono::high_resolution_clock::now();

            std::vector<InputType> inputs = window.input();

            for (InputType input : inputs) {
                //Better switch because switch does not work for variants. It does look hard on the eye but it work similar to a switch
                std::visit([this](auto&& arg) {
                    using T = std::decay_t<decltype(arg)>;

                    if constexpr (std::is_same_v<T, Mouse::LeftClick>) {
                        std::cout << "Mouse Left Click at (" << arg.position.x << ", " << arg.position.y << ")\n"; //TODO force (the longer you hold the bigger the force becomes)
                    } 
                    else if constexpr (std::is_same_v<T, Mouse::RightClick>) {
                        std::cout << "Mouse Right Click at (" << arg.position.x << ", " << arg.position.y << ")\n";
                    }
                    else if constexpr (std::is_same_v<T, Mouse::MiddleClick>) {
                        std::cout << "Mouse Middle Click at (" << arg.position.x << ", " << arg.position.y << ")\n";
                        addParticle(arg.position);
                    }
                    else if constexpr (std::is_same_v<T, Keyboard::KeyPressed>) {
                        std::cout << "Keyboard Key Pressed: " << static_cast<char>(arg.keyCode) << " (code: " << arg.keyCode << ")\n";
                    }
                    else if constexpr (std::is_same_v<T, Keyboard::KeyReleased>) {
                        std::cout << "Keyboard Key Released: " << static_cast<char>(arg.keyCode) << " (code: " << arg.keyCode << ")\n";
                    }
                    else if constexpr (std::is_same_v<T, Keyboard::KeyHeld>) {
                        std::cout << "Keyboard Key Held: " << arg.keyCode << " (code: " << static_cast<int>(arg.keyCode) << ")\n";
                        this->particles.clear();
                    }
                }, input);
            }

            std::cout << "--- Input Processing Finished ---\n";

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

    void Simulation::addForce(const vector2& pos) {
        for (Particle particle : particles) {
            particle.applyForce(pos, 10);
        }
    }
    void Simulation::removeForce(const vector2& pos) {
        for (Particle particle : particles) {
            particle.applyForce(pos, -10);
        }
    }
    void Simulation::addParticle(const vector2& pos) {
        particles.push_back(Particle(1.0, 0.1, pos));
    }

    