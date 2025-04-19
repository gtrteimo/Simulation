#include "gui/frame.hpp"

static void inline resize([[maybe_unused]] GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

Frame::Frame() : Frame(1000, 1000){}
Frame::Frame(uint16_t frameWidth, uint16_t frameHeight) : frameWidth(frameWidth), frameHeight(frameHeight){
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
	}
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  #endif
    window = glfwCreateWindow(frameWidth, frameHeight, "OpenGL", NULL, NULL);
    if (!window) {
        glfwTerminate();
		throw std::runtime_error("Failed to create window");
	}
	glfwMakeContextCurrent(window);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		glfwTerminate();
		throw std::runtime_error("Failed to initialize GLAD");
	}

	glViewport(0, 0, frameWidth, frameHeight);

    glfwSetFramebufferSizeCallback(window, resize);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}
Frame::~Frame() {
    glfwTerminate();
}

int Frame::update() {
	glClear(GL_COLOR_BUFFER_BIT);

	// processInput(window);

	glfwSwapBuffers(window);
	glfwPollEvents();
	return 0;
}