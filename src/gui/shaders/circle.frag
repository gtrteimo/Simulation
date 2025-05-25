#version 460 core
out vec4 FragColor;
uniform vec3 u_circleColor;

void main() {
    FragColor = vec4(u_circleColor, 1.0);
}