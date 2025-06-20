#include "gui/frame.hpp"
#include <iostream>

void Frame::framebuffer_size_callback(GLFWwindow* cb_window, int width, int height) {
    (void)cb_window;
    glViewport(0, 0, width, height);
}

Frame::Frame(uint16_t width, uint16_t height, const char* title)
    : window(nullptr), frameWidth(width), frameHeight(height) {

    window = glfwCreateWindow(frameWidth, frameHeight, title, NULL, NULL);
    if (!window) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);

    static bool glad_initialized = false;
    if (!glad_initialized) {
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwDestroyWindow(window);
            throw std::runtime_error("Failed to initialize GLAD");
        }
        glad_initialized = true;
    }

    glViewport(0, 0, frameWidth, frameHeight);
    glfwSetFramebufferSizeCallback(window, Frame::framebuffer_size_callback);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
}

Frame::~Frame() {
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
}

GLFWwindow *Frame::getWindow() const {
    return window;
}

uint16_t Frame::getFrameWidth() const {
    return frameWidth;
}
uint16_t Frame::getFrameHeight() const {
    return frameHeight;
}


