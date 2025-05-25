#version 460 core
layout (location = 0) in vec2 aPos; // Vertex positions

// If you want to pass center and radius via uniforms for a unit circle mesh:
// uniform vec2 u_offset; // (ndc_center_x, ndc_center_y)
// uniform vec2 u_scale;  // (ndc_radius_x, ndc_radius_y)

void main() {
    // If vertices are pre-transformed:
    gl_Position = vec4(aPos.x, aPos.y, 0.0, 1.0);

    // If using a unit circle and transforming here:
    // gl_Position = vec4(aPos.x * u_scale.x + u_offset.x, aPos.y * u_scale.y + u_offset.y, 0.0, 1.0);
}