#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <stdexcept>
#include <vector>

#include "util/types.h"
#include "util/vector3.hpp" // Assuming this defines vector3 and type
// If 'type' is not defined in vector3.hpp, ensure it's defined (e.g., typedef double type;)

class Frame {
  private:
    GLFWwindow *window;
    uint16_t frameWidth;
    uint16_t frameHeight;

    // Private static callback (or free function)
    static void framebuffer_size_callback(GLFWwindow *window, int width, int height);

  public:
    Frame(uint16_t width, uint16_t height, const char* title = "OpenGL");
    ~Frame();

    // Make Frame non-copyable and non-movable to enforce single ownership
    // Or implement proper move semantics if moving a Frame object is desired.
    Frame(const Frame&) = delete;
    Frame& operator=(const Frame&) = delete;
    Frame(Frame&&) = delete; // For simplicity, disable move too. Implement if needed.
    Frame& operator=(Frame&&) = delete; // For simplicity, disable move too. Implement if needed.

    GLFWwindow *getWindow() const; // Make it const
    uint16_t getFrameWidth() const;
    uint16_t getFrameHeight() const;

    // Input and Update methods
    int input(std::vector<vector3> &ret);
    // bool shouldClose() const;
};