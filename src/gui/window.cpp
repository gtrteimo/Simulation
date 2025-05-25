
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdio.h>

static inline void framebuffer_size_callback([[maybe_unused]] GLFWwindow *window, int width, int height) {
	glViewport(0, 0, width, height);
}

int createWindow() {
	if (!glfwInit()) {
		fprintf(stderr, "Failed to initialize GLFW\n");
		fflush(stderr);
		return -1;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
#endif

	GLFWwindow *window = glfwCreateWindow(1000, 1000, "OpenGL", NULL, NULL);

	if (!window) {
		fprintf(stderr, "Failed to create window\n");
		fflush(stderr);
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		fprintf(stderr, "Failed to initialize GLAD\n");
		fflush(stderr);
		return -1;
	}

	glViewport(0, 0, 720, 720);

	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

	glClearColor(255, 255, 255, 1.0f);

	while (!glfwWindowShouldClose(window)) {
		glClear(GL_COLOR_BUFFER_BIT);

		glfwSwapBuffers(window);
		glfwPollEvents();
	}

	glfwTerminate();

	return 0;
}
