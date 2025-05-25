#include "gui/frame.hpp"
#include <iostream> // For MouseClick logging if re-enabled

// Callback function needs to be static or a free function
void Frame::framebuffer_size_callback(GLFWwindow* cb_window, int width, int height) {
    (void)cb_window; // Unused parameter
    glViewport(0, 0, width, height);
    // Potentially, update Frame's stored width/height if it's associated with cb_window
    // For that, you might need glfwSetWindowUserPointer and glfwGetWindowUserPointer
}

// Removed default constructor Frame() : Frame(1000, 1000) {}
// If needed, add it back but ensure it calls the main constructor.
Frame::Frame(uint16_t width, uint16_t height, const char* title)
    : window(nullptr), frameWidth(width), frameHeight(height) {

    // GLFW window hints should be set BEFORE window creation
    // These are now expected to be set by the main application before creating a Frame
    // or set them here if they are specific to this Frame type.
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    // glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    // glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    // #ifdef __APPLE__
    // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    // #endif

    window = glfwCreateWindow(frameWidth, frameHeight, title, NULL, NULL);
    if (!window) {
        // glfwTerminate(); // Don't call terminate here, main app handles it
        throw std::runtime_error("Failed to create GLFW window");
    }
    glfwMakeContextCurrent(window);

    // GLAD should be initialized AFTER a context is current.
    // If multiple Frames might exist, GLAD only needs to be loaded once.
    // This is best done in the main application after the first window/context.
    // For simplicity here, we assume this Frame is the one initializing GLAD.
    // A more robust solution would check if GLAD is already loaded.
    static bool glad_initialized = false;
    if (!glad_initialized) {
        if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
            glfwDestroyWindow(window); // Clean up the window we just made
            // glfwTerminate(); // Main app handles termination
            throw std::runtime_error("Failed to initialize GLAD");
        }
        glad_initialized = true;
    }


    // int fbW, fbH;
    // glfwGetFramebufferSize(window, &fbW, &fbH);
    // glViewport(0, 0, fbW, fbH);

    glViewport(0, 0, frameWidth, frameHeight);
    glfwSetFramebufferSizeCallback(window, Frame::framebuffer_size_callback);
    // Store 'this' pointer if callback needs to update Frame's width/height
    // glfwSetWindowUserPointer(window, this); 

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Default clear color
}

Frame::~Frame() {
    if (window) {
        glfwDestroyWindow(window);
        window = nullptr;
    }
    // DO NOT CALL glfwTerminate() here. Main application will do it.
}

GLFWwindow *Frame::getWindow() const { // Marked const
    return window;
}

uint16_t Frame::getFrameWidth() const {
    return frameWidth;
}
uint16_t Frame::getFrameHeight() const {
    return frameHeight;
}

bool Frame::shouldClose() const {
    if (!window) return true;
    return glfwWindowShouldClose(window);
}

// Helper free functions (could be static members or in a utility namespace)
// Note: 'type' needs to be defined (e.g. typedef double type;)
vector3 MouseClick(GLFWwindow *window, int button) { // Combined helper
    vector3 ret = vector3(-2, -2, -2); // Assuming vector3 constructor and 'type'
    if (glfwGetMouseButton(window, button) == GLFW_PRESS) {
        double x, y;
        glfwGetCursorPos(window, &x, &y);
        // If your 0-1 coordinate system has Y increasing upwards:
        // int current_height;
        // glfwGetFramebufferSize(window, nullptr, ¤t_height);
        // y = static_cast<double>(current_height) - y;
        ret = vector3(static_cast<type>(x), static_cast<type>(y), 0.0);
    }
    return ret;
}


int Frame::input(std::vector<vector3> &ret) {
    ret.assign(2, vector3(-2, -2, -2)); // Resize and initialize

    if (!window) return -2; // Window doesn't exist

    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
        glfwSetWindowShouldClose(window, GLFW_TRUE);
        return -1; // Indicate app should close
    }

    vector3 leftClickPos = MouseClick(window, GLFW_MOUSE_BUTTON_LEFT);
    if (leftClickPos.x >= 0) { // Check if click occurred
        // Normalize based on current framebuffer size for accuracy if window is resized
        int current_fb_width, current_fb_height;
        glfwGetFramebufferSize(window, &current_fb_width, &current_fb_height);
        if (current_fb_width > 0 && current_fb_height > 0) {
			// std::cout << "width: " << current_fb_width << ", height: " << current_fb_height << std::endl;
            // Assuming (0,0) is bottom-left for your normalized coords
            // and glfwGetCursorPos gives (0,0) at top-left
            type normalized_x = static_cast<type>(leftClickPos.x) / static_cast<type>(current_fb_width) * 2.0 - 1;
            type normalized_y = 1.0 - static_cast<type>(static_cast<type>(leftClickPos.y) / static_cast<type>(current_fb_height)) * 2.0; // Flip Y

            ret[0] = vector3(normalized_x, normalized_y, 0.0);
            // if (ret[0].isValid()) { /* process */ } // Assuming isValid checks 0-1 range
        }
    }

    vector3 rightClickPos = MouseClick(window, GLFW_MOUSE_BUTTON_RIGHT);
    if (rightClickPos.x >= 0) {
        int current_fb_width, current_fb_height;
        glfwGetFramebufferSize(window, &current_fb_width, &current_fb_height);
         if (current_fb_width > 0 && current_fb_height > 0) {
            type normalized_x = static_cast<type>(rightClickPos.x) / static_cast<type>(current_fb_width) * 2.0 - 1;
            type normalized_y = 1.0 - (static_cast<type>(rightClickPos.y) / static_cast<type>(current_fb_height)) * 2.0; // Flip Y
            ret[1] = vector3(normalized_x, normalized_y, 0.0);
            // if (ret[1].isValid()) { /* process */ }
        }
    }
    return 0;
}

void Frame::update() { // Changed to void
    if (!window) return;
    // Clear screen is often part of the rendering pass, not generic frame update.
    // If Draw class also clears, this might be redundant or intentional.
    // glClear(GL_COLOR_BUFFER_BIT); //  Moved to Draw::clearScreen or main render loop

    glfwSwapBuffers(window);
    glfwPollEvents();
    // return 0;
}