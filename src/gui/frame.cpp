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

// vector2 MouseClick(GLFWwindow *window, int button) {
//     vector2 ret = vector2(-2, -2;
//     if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
//         double x, y;
//         glfwGetCursorPos(window, &x, &y);
//         ret = vector2(static_cast<type>(x), static_cast<type>(y));
//     }
//     return ret;
// }
vector2 normalize(GLFWwindow* const window, const vector2& vec) {
        int current_fb_width, current_fb_height;
        glfwGetFramebufferSize(window, &current_fb_width, &current_fb_height);
        if (current_fb_width > 0 && current_fb_height > 0) {
            type normalized_x = static_cast<type>(vec.x) / static_cast<type>(current_fb_width) * 2.0 - 1;
            type normalized_y = 1.0 - static_cast<type>(static_cast<type>(vec.y) / static_cast<type>(current_fb_height)) * 2.0;
            return {normalized_x, normalized_y};
        }
        return {-1, -1};
}

std::vector<InputType> Frame::input() {

    if (!window) {
        return {};
    } 

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return {};
    }

    std::vector<InputType> ret = std::vector<InputType>();

    double x, y;
    glfwGetCursorPos(window, &x, &y);
    
    if (x >= 0 && y >= 0) {
        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS) {
            vector2 position = (normalize(window, {static_cast<type>(x), static_cast<type>(y)}));
            if (position.isValid()) {
                ret.push_back(Mouse::LeftClick{position});
            }
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
            vector2 position = (normalize(window, {static_cast<type>(x), static_cast<type>(y)}));
            if (position.isValid()) {
                ret.push_back(Mouse::RightClick{position});
            }
        }

        if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
            vector2 position = (normalize(window, {static_cast<type>(x), static_cast<type>(y)}));
            if (position.isValid()) {
                ret.push_back(Mouse::MiddleClick{position});
            }
        }

        if (glfwGetKey(window, GLFW_KEY_C) == GLFW_PRESS) {
            ret.push_back(Keyboard::KeyHeld{'C'});
        }
    }

    return ret;
}
