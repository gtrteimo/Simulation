#include "simulation/draw.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

const char *vertexShaderSource = R"glsl(
    #version 460 core
    layout (location = 0) in vec2 aPos;
    void main() {
        gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);
    }
)glsl";

const char *fragmentShaderSource = R"glsl(
    #version 460 core
    out vec4 FragColor;
    uniform vec3 u_Color;
    void main() {
        FragColor = vec4(u_Color, 1.0);
    }
)glsl";

// --- Shader Compilation Helpers ---
GLuint Draw::compileShader(GLenum shaderType, const char *source) {
	GLuint shader = glCreateShader(shaderType);
	glShaderSource(shader, 1, &source, nullptr);
	glCompileShader(shader);

	GLint success;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetShaderInfoLog(shader, 512, nullptr, infoLog);
		std::cerr << "ERROR::SHADER::COMPILATION_FAILED\nType: " << shaderType << "\n"
		          << infoLog << std::endl;
		glDeleteShader(shader);
		return 0;
	}
	return shader;
}

GLuint Draw::linkShaderProgram(GLuint vertexShader, GLuint fragmentShader) {
	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);
	glLinkProgram(program);

	GLint success;
	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		char infoLog[512];
		glGetProgramInfoLog(program, 512, nullptr, infoLog);
		std::cerr << "ERROR::SHADER::PROGRAM::LINKING_FAILED\n"
		          << infoLog << std::endl;
		glDeleteProgram(program);
		return 0;
	}
	return program;
}

Draw::Draw(Frame &existingFrame) : window(existingFrame.getWindow()), shaderProgramID(0), shapeVAO(0), shapeVBO(0) {
	if (!this->window) { // Use this->window to be explicit
		throw std::runtime_error("Failed to get GLFW window from Frame for Draw object");
	}

	// Compile and link shaders (same as before)
	GLuint vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
	GLuint fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

	if (vertexShader == 0 || fragmentShader == 0) {
		throw std::runtime_error("Failed to compile shaders.");
	}

	shaderProgramID = linkShaderProgram(vertexShader, fragmentShader);
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	if (shaderProgramID == 0) {
		throw std::runtime_error("Failed to link shader program.");
	}

	// Setup VAO and VBO for shapes (same as before)
	glGenVertexArrays(1, &shapeVAO);
	glGenBuffers(1, &shapeVBO);

	glBindVertexArray(shapeVAO);
	glBindBuffer(GL_ARRAY_BUFFER, shapeVBO);
	glVertexAttribPointer(0, 2, GL_DOUBLE, GL_FALSE, 2 * sizeof(type), (void *)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

Draw::~Draw() {
	if (shaderProgramID != 0) {
		glDeleteProgram(shaderProgramID);
	}
	if (shapeVBO != 0) {
		glDeleteBuffers(1, &shapeVBO);
	}
	if (shapeVAO != 0) {
		glDeleteVertexArrays(1, &shapeVAO);
	}
}

int Draw::drawParticles(const std::vector<Particle> &particles) {
	for (const Particle &particle : particles) {
		drawCircle(particle.position, particle.size, particle.colour, true, 32);
		// drawRectangle(particle.position, particle.size, particle.size, particle.colour, true);

	}
	return 0;
}

int Draw::drawParticle(const Particle &particle) {
	return drawCircle(particle.position, particle.size, particle.colour, true, 32);
}

int Draw::drawLine(const vector2 &start, const vector2 &end, const colourRGB &colour) {
	if (!window || shaderProgramID == 0) return -1;

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
	if (screenHeight == 0 || screenWidth == 0) return -2;

	// Perform calculations with doubles (implicit from type), then cast to type
	type ndc_start_x = static_cast<type>(start.x * 2.0 - 1.0);
	type ndc_start_y = static_cast<type>(start.y * 2.0 - 1.0);
	type ndc_end_x = static_cast<type>(end.x * 2.0 - 1.0);
	type ndc_end_y = static_cast<type>(end.y * 2.0 - 1.0);

	vertex_buffer_data.clear(); // Assuming vertex_buffer_data is std::vector<type>
	vertex_buffer_data.push_back(ndc_start_x);
	vertex_buffer_data.push_back(ndc_start_y);
	vertex_buffer_data.push_back(ndc_end_x);
	vertex_buffer_data.push_back(ndc_end_y);

	glUseProgram(shaderProgramID);
	GLint colorLocation = glGetUniformLocation(shaderProgramID, "u_Color");
	if (colorLocation != -1) { // Good practice to check if uniform is found
		glUniform3f(colorLocation, static_cast<float>(colour.r)/255.0f, static_cast<float>(colour.g)/255.0f, static_cast<float>(colour.b)/255.0f);
	} else {
		std::cerr << "Warning: u_Color uniform not found in shader program " << shaderProgramID << std::endl; // ADD THIS
	}

	glBindVertexArray(shapeVAO);
	glBindBuffer(GL_ARRAY_BUFFER, shapeVBO);
	glBufferData(GL_ARRAY_BUFFER,
	             static_cast<GLsizeiptr>(vertex_buffer_data.size() * sizeof(type)),
	             vertex_buffer_data.data(),
	             GL_DYNAMIC_DRAW);

	glDrawArrays(GL_LINES, 0, 2);

	glBindVertexArray(0);
	glUseProgram(0);
	return 0;
}

int Draw::drawCircle(const vector2 center, type radius, const colourRGB &colour, bool filled, int segments) {
	if (!window || shaderProgramID == 0) return -1;
	if (segments < 3) segments = 3;

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
	if (screenHeight == 0 || screenWidth == 0) return -2;

	// Convert center and radius from pixel space to NDC
	type ndc_center_x = static_cast<type>((center.x));
	type ndc_center_y = static_cast<type>((center.y));
	type ndc_radius_x = static_cast<type>((radius));
	type ndc_radius_y = static_cast<type>((radius));

	// std::cout << "NDC Center: (" << ndc_center_x << ", " << ndc_center_y
	//           << "), Radius: (" << ndc_radius_x << ", " << ndc_radius_y << ")" << std::endl;

	vertex_buffer_data.clear();
	if (filled) {
		vertex_buffer_data.push_back(ndc_center_x);
		vertex_buffer_data.push_back(ndc_center_y);
	}

	for (int i = 0; i <= segments; ++i) {
		type angle_rad = static_cast<type>(2.0) * static_cast<type>(M_PI) * static_cast<type>(i) / static_cast<type>(segments);
		type x = ndc_center_x + ndc_radius_x * std::cos(angle_rad);
		type y = ndc_center_y + ndc_radius_y * std::sin(angle_rad);
		vertex_buffer_data.push_back(x);
		vertex_buffer_data.push_back(y);
	}

	glUseProgram(shaderProgramID);
	GLint colorLocation = glGetUniformLocation(shaderProgramID, "u_Color");
	if (colorLocation != -1) {
		glUniform3f(colorLocation, static_cast<float>(colour.r)/255.0f, static_cast<float>(colour.g)/255.0f, static_cast<float>(colour.b)/255.0f);
	} else {
		std::cerr << "Warning: u_Color uniform not found in shader program " << shaderProgramID << std::endl; // ADD THIS
	}

	glBindVertexArray(shapeVAO);
	glBindBuffer(GL_ARRAY_BUFFER, shapeVBO);
	glBufferData(GL_ARRAY_BUFFER,
	             static_cast<GLsizeiptr>(vertex_buffer_data.size() * sizeof(type)),
	             vertex_buffer_data.data(),
	             GL_DYNAMIC_DRAW);

	if (filled) {
		glDrawArrays(GL_TRIANGLE_FAN, 0, static_cast<GLsizei>(segments + 2));
	} else {
		glDrawArrays(GL_LINE_LOOP, 0, static_cast<GLsizei>(segments + 1));
	}

	glBindVertexArray(0);
	glUseProgram(0);
	return 0;
}

int Draw::drawRectangle(const vector2 &position, type width, type height, const colourRGB &colour, bool filled) {
	if (!window || shaderProgramID == 0) return -1;

	int screenWidth, screenHeight;
	glfwGetFramebufferSize(window, &screenWidth, &screenHeight);
	if (screenHeight == 0 || screenWidth == 0) return -2;

	type ndc_x = static_cast<type>(position.x * 2.0 - 1.0);
	type ndc_y = static_cast<type>(position.y * 2.0 - 1.0);
	// width and height are 'type' (double)
	type ndc_w = static_cast<type>(static_cast<double>(width) * 2.0);
	type ndc_h = static_cast<type>(static_cast<double>(height) * 2.0);

	type x0 = ndc_x;
	type y0 = ndc_y;
	type x1 = ndc_x + ndc_w;
	type y1 = ndc_y + ndc_h;

	vertex_buffer_data.clear(); // Assuming vertex_buffer_data is std::vector<type>

	if (filled) {
		vertex_buffer_data.push_back(x0);
		vertex_buffer_data.push_back(y0);
		vertex_buffer_data.push_back(x1);
		vertex_buffer_data.push_back(y0);
		vertex_buffer_data.push_back(x0);
		vertex_buffer_data.push_back(y1);

		vertex_buffer_data.push_back(x1);
		vertex_buffer_data.push_back(y0);
		vertex_buffer_data.push_back(x1);
		vertex_buffer_data.push_back(y1);
		vertex_buffer_data.push_back(x0);
		vertex_buffer_data.push_back(y1);
	} else {
		vertex_buffer_data.push_back(x0);
		vertex_buffer_data.push_back(y0);
		vertex_buffer_data.push_back(x1);
		vertex_buffer_data.push_back(y0);
		vertex_buffer_data.push_back(x1);
		vertex_buffer_data.push_back(y1);
		vertex_buffer_data.push_back(x0);
		vertex_buffer_data.push_back(y1);
	}

	glUseProgram(shaderProgramID);
	GLint colorLocation = glGetUniformLocation(shaderProgramID, "u_Color");
	if (colorLocation != -1) {
		glUniform3f(colorLocation, static_cast<float>(colour.r)/255.0f, static_cast<float>(colour.g)/255.0f, static_cast<float>(colour.b)/255.0f);
	} else {
		std::cerr << "Warning: u_Color uniform not found in shader program " << shaderProgramID << std::endl; // ADD THIS
	}

	glBindVertexArray(shapeVAO);
	glBindBuffer(GL_ARRAY_BUFFER, shapeVBO);
	glBufferData(GL_ARRAY_BUFFER,
	             static_cast<GLsizeiptr>(vertex_buffer_data.size() * sizeof(type)),
	             vertex_buffer_data.data(),
	             GL_DYNAMIC_DRAW);

	if (filled) {
		glDrawArrays(GL_TRIANGLES, 0, 6);
	} else {
		glDrawArrays(GL_LINE_LOOP, 0, 4);
	}

	glBindVertexArray(0);
	glUseProgram(0);
	return 0;
}

int Draw::drawText(const std::string &text, const vector2 &position, const colourRGB &colour) {
	// To avoid unused parameter warnings if you don't implement it yet:
	(void)text;
	(void)position;
	(void)colour;
	std::cerr << "Warning: Draw::drawText is not implemented." << std::endl;
	return -1;
}

int Draw::clearScreen(const colourRGB &clear_colour) {
	if (!window) return -1;
	glClearColor(clear_colour.r, clear_colour.g, clear_colour.b, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	return 0;
}

int Draw::swapBuffers() {
	if (!window) return -1;
	glfwSwapBuffers(window);
	// No need to call glfwPollEvents() here, that's usually in the main loop
	return 0;
}