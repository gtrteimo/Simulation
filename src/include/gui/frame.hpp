#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <iostream>

#include "util/vector3.hpp"
#include "util/colour.h"

class Frame {
  private:
    uint16_t frameWidth;
	  uint16_t frameHeight;
    GLFWwindow *window;
  public:
    Frame();
    Frame(uint16_t frameWidth, uint16_t frameHeight);
	  ~Frame();

    GLFWwindow* getWindow();

    int input();
    int update();

    
  private:

};