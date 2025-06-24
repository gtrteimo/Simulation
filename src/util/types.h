#pragma once

#include <inttypes.h>

typedef double type; 
/*
 * Carful with this, there are some places where changing it to float will break things with the OpenGL code. 
 * For example in draw.cpp: 	"glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 2 * sizeof(type), (void *)0);"    where you tell OpenGL that the data is of type GL_DOUBLE. 
 */