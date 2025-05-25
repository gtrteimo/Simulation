#pragma once

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <stdexcept>
#include <iostream>
#include <cmath> 

#include "util/vector3.hpp"
#include "util/colour.h"
#include "simulation/particle.hpp"
#include "gui/frame.hpp"

class Draw {
  private:
    GLFWwindow *window;

    GLuint shaderProgramID;
    GLuint shapeVAO;
    GLuint shapeVBO;

    static GLuint compileShader(GLenum shaderType, const char* source);
    static GLuint linkShaderProgram(GLuint vertexShader, GLuint fragmentShader);
    
    std::vector<type> vertex_buffer_data;

  public:
    Draw(Frame& existingFrame);
    ~Draw();

    int drawParticles(const std::vector<Particle> &particles);
    int drawParticle(const Particle &particle);
    int drawLine(const vector3 &start, const vector3 &end, const colourRGB &colour);
    int drawCircle(const vector3 center, type radius, const colourRGB &colour, bool filled = true, int segments = 36);
    int drawRectangle(const vector3 &position, type width, type height, const colourRGB &colour, bool filled = true);
    int drawText(const std::string &text, const vector3 &position, const colourRGB &colour); // TODO or maybee not
    
    int clearScreen(const colourRGB &clear_colour = {0, 0, 0});
    int swapBuffers();
};