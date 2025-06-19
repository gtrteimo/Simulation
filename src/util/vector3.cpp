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

bool vector3::isValid() const {
    bool ret = true;
    if (x < -1 || y < -1 || z < -1) {
        ret = false;
    }
    if (x > 1 || y > 1 || z > 1) {
        ret = false;
    }
    return ret;
}

void vector3::empty() {
    x = 0;
    y = 0;
    z = 0;
}