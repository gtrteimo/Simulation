#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

static bool s = false;

static void inline framebuffer_size_callback([[maybe_unused]] GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

static void inline processInput([[maybe_unused]] GLFWwindow *window) {
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	if (glfwGetKey(window, GLFW_KEY_ENTER) == GLFW_PRESS) {
		if (s) {
			glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		} else {
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
		}
		s = !s;
	}
}

int createWindow() {
	if (!glfwInit()) {
		std::cerr << "Failed to initialize GLFW" << std::endl;
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

	GLFWwindow *window = glfwCreateWindow(720, 720, "OpenGL", NULL, NULL);

	if (!window) {
		std::cerr << "Failed to create window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return -1;
	}

	glViewport(0, 0, 720, 720);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		processInput(window);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}
