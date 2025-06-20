
#include "simulation/simulation.hpp"

static int64_t leftMouseButtonHeld = 0;
static int64_t rightMouseButtonHeld = 0;
static uint64_t middleMouseButtonHeld = 0;

Simulation* Simulation::sim = nullptr;

Simulation::Simulation(uint16_t fps, uint64_t particleAmount) : Simulation(fps, 1000, 1000, particleAmount) {}
Simulation::Simulation(uint16_t fps, uint16_t frameWidth, uint16_t frameHeight, uint64_t particleAmount) : fps(fps + 1), window(Frame(frameWidth, frameHeight)), draw(Draw(window)), particles(std::vector<Particle>(particleAmount, Particle(1.0))) {sim = this;}

Simulation::~Simulation() {
	particles.clear();
	particles.shrink_to_fit();
}

inline int Simulation::updateAccVel() {
	if (fps == 0) {
		return -1;
	}
	type dt = static_cast<type>(1.0) / static_cast<type>(fps);
	for (Particle &particle : particles) {
		particle.updateAccVel(dt);
	}
	return 0;
}

inline int Simulation::updatePos() {
	if (fps == 0) {
		return -1;
	}
	type dt = static_cast<type>(1.0) / static_cast<type>(fps);
	for (Particle &particle : particles) {
		particle.updatePos(dt);
	}
	return 0;
}

vector2 normalize(GLFWwindow *const window, const vector2 &vec) {
	int current_fb_width, current_fb_height;
	glfwGetFramebufferSize(window, &current_fb_width, &current_fb_height);
	if (current_fb_width > 0 && current_fb_height > 0) {
		type normalized_x = static_cast<type>(vec.x) / static_cast<type>(current_fb_width) * 2.0 - 1;
		type normalized_y = 1.0 - static_cast<type>(static_cast<type>(vec.y) / static_cast<type>(current_fb_height)) * 2.0;
		return {normalized_x, normalized_y};
	}
	return {-1, -1};
}

int Simulation::loop(std::function<int(std::vector<Particle>)> function) {

	// Duration of one frame in nanoseconds
	const std::chrono::nanoseconds frameTime(static_cast<long long>(1e9 / fps));

    glfwSetMouseButtonCallback(window.getWindow(), inputMouseTap);
    glfwSetKeyCallback(window.getWindow(), inputKeyboardTap);

	while (!glfwWindowShouldClose(window.getWindow())) {

		// Timer start
		auto start = std::chrono::high_resolution_clock::now();

        if (leftMouseButtonHeld > 0) {
            double x, y;
	        glfwGetCursorPos(window.getWindow(), &x, &y);
            if (x >= 0 && y >= 0) {
                vector2 position = (normalize(window.getWindow(), {static_cast<type>(x), static_cast<type>(y)}));
                if (position.isValid()) {
                    std::cout << "Mouse Left Held at (" << position.x << ", " << position.y << ")\n"; // TODO force (the longer you hold the bigger the force becomes)
                    addForce(position);
                }
            }

            leftMouseButtonHeld++;
        }
        if (rightMouseButtonHeld > 0) {
            double x, y;
	        glfwGetCursorPos(window.getWindow(), &x, &y);
            if (x >= 0 && y >= 0) {
                vector2 position = (normalize(window.getWindow(), {static_cast<type>(x), static_cast<type>(y)}));
                if (position.isValid()) {
                    std::cout << "Mouse Right Held at (" << position.x << ", " << position.y << ")\n"; // TODO force (the longer you hold the bigger the force becomes)
                    removeForce(position);
                }
            }

            rightMouseButtonHeld++;
        }
        if (middleMouseButtonHeld > 0) {
            double x, y;
	        glfwGetCursorPos(window.getWindow(), &x, &y);
            if (x >= 0 && y >= 0) {
                vector2 position = (normalize(window.getWindow(), {static_cast<type>(x), static_cast<type>(y)}));
                if (position.isValid()) {
                    std::cout << "Mouse Middle Held at (" << position.x << ", " << position.y << ")\n"; // TODO force (the longer you hold the bigger the force becomes)
                    for (Particle particle : particles) {
                        particle.printParticle();
                    }
                    addParticle(position, 1);
                }
            }

            middleMouseButtonHeld++;
        }

		if (int ret = (function(particles)) < 0) {
			return ret;
		}

        updateAccVel();
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

void Simulation::inputMouseTap(GLFWwindow* window, int button, int action, [[maybe_unused]] int mods) {

    if (!window) {
		return;
	}

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            leftMouseButtonHeld = 1;
        }
        else if (action == GLFW_RELEASE) {
            leftMouseButtonHeld = 0;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            rightMouseButtonHeld = 1;
        }
        else if (action == GLFW_RELEASE) {
            rightMouseButtonHeld = 0;
        } 
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            middleMouseButtonHeld = 1;
        }
        else if (action == GLFW_RELEASE) {
            middleMouseButtonHeld = 0;
        }
    }

}
void Simulation::inputKeyboardTap(GLFWwindow* window, int key, [[maybe_unused]] int scancode, int action, [[maybe_unused]] int mods) {

    if (!window) {
		return;
	}

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
		return;
	}

    if (key == GLFW_KEY_C && (action == GLFW_PRESS || action == GLFW_REPEAT)) {
        std::cout << "Key C pressed down\n";
        sim->particles.clear();
    }
}

void Simulation::addForce(const vector2 &pos) {
	for (Particle particle : particles) {
		particle.applyForce(pos);
	}
}
void Simulation::removeForce(const vector2 &pos) {
	for (Particle particle : particles) {
		particle.applyForce(-pos);
	}
}
void Simulation::addParticle(const vector2 &pos, uint64_t amount) {
    for (uint64_t i = 0; i < amount; i++) {
	    particles.push_back(Particle(1.0, 0.05, pos));
    }
}
