#pragma once

#include "util/types.h"
#include <iostream>

struct vector2 {
    type x;
	type y;

    vector2() : x(0), y(0) {}

    vector2(type x_val, type y_val) : x(x_val), y(y_val) {}

    vector2 operator+(const vector2& other) const;

    vector2 operator-(const vector2& other) const;

    vector2 operator*(const type scalar) const;

    type dot(const vector2& other) const;

    vector2 operator/(const type scalar) const;

    vector2& operator+=(const vector2& other);

    vector2& operator-=(const vector2& other);

	vector2& operator*=(const type scalar);

	vector2& operator/=(const type scalar);

    vector2 operator-() const;

    bool isValid() const;

    bool isEmpty();

    void empty();
};

std::ostream& operator<<(std::ostream& os, const vector2& vec);