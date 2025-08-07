#include "util/vector2.hpp"


vector2 vector2::operator+(const vector2& other) const{
    return {x + other.x, y + other.y};
}

vector2 vector2::operator-(const vector2& other) const {
    return {x - other.x, y - other.y};
}

vector2 vector2::operator*(type scalar) const {
    return {x * scalar, y * scalar};
}

type vector2::dot(const vector2& other) const {
    return x * other.x + y * other.y;
}

vector2 vector2::operator/(const type scalar) const {
     if (scalar == 0) {
        std::cerr << "Warning: Division by zero in vector2::operator/(type scalar)\n";
        return vector2(0, 0);
    }
    return {x / scalar, y / scalar};
}

vector2& vector2::operator+=(const vector2& other) {
    x += other.x;
    y += other.y;
    return *this;
}

vector2& vector2::operator-=(const vector2& other) {
    x -= other.x;
    y -= other.y;
    return *this;
}

vector2& vector2::operator*=(const type scalar) {
    x *= scalar;
    y *= scalar;
    return *this;
}

vector2& vector2::operator/=(const type scalar) {
    x /= scalar;
    y /= scalar;
    return *this;
}

vector2 vector2::operator-() const {
    return {-x, -y};
}

std::ostream& operator<<(std::ostream &os, const vector2 &vec) {
	os << "(x: " << vec.x << ", y: " << vec.y << ")";
    return os;
}

type vector2::magnitudeSq() const {
    return x * x + y * y;
}

type vector2::magnitude() const {
    return std::sqrt(magnitudeSq());
}

vector2 vector2::normalize() const {
    type mag = magnitude();
    if (mag == static_cast<type>(0.0)) {
        // Handle zero-length vector gracefully (e.g., return zero vector or throw error)
        std::cerr << "Trying to normalize a vector2 what has a magintude/length of 0\n";
        return {static_cast<type>(0.0), static_cast<type>(0.0)}; // For collision, if length is zero, centers are coincident.
    }
    return {x / mag, y / mag};
}

bool vector2::isValid() const {
    bool ret = true;
    if (x < -1 || y < -1) {
        ret = false;
    }
    if (x > 1 || y > 1) {
        ret = false;
    }
    return ret;
}

bool vector2::isEmpty() {
    if ( !x && !y) {
        return true;
    }
    return false;
}

void vector2::empty() {
    x = 0;
    y = 0;
}