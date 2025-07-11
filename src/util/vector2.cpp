#include "util/vector2.hpp"


vector2 vector2::operator+(const vector2& other) const{
    return {x + other.x, y + other.y};
}

vector2 vector2::operator-(const vector2& other) const {
    return {x - other.x, y + other.y};
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