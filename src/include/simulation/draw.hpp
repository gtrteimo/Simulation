#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>

#include "util/vector3.hpp"
#include "util/colour.h"
#include "simulation/particle.hpp"

class Draw {
  private: 
    GLFWwindow *window;
    std::vector<Particle> particles;
  public:
  	Draw(GLFWwindow *window);
  	
};