#include "util/vector3.hpp"


vector3 vector3::operator+(const vector3& other) const{
    return {x + other.x, y + other.y, z + other.z};
}

vector3 vector3::operator*(type scalar) const {
    return {x * scalar, y * scalar, z * scalar};
}

vector3& vector3::operator+=(const vector3& other) {
    x += other.x;
    y += other.y;
    z += other.z;
    return *this;
}

vector3& vector3::operator*=(type scalar) {
    x *= scalar;
    y *= scalar;
    z *= scalar;
    return *this;
}