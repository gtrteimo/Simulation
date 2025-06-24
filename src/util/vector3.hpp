#pragma once

#include "util/types.h"
#include <iostream>

struct vector3 {
    type x;
	type y;
	type z;

    vector3() : x(0), y(0), z(0) {}

    vector3(type x_val, type y_val, type z_val) : x(x_val), y(y_val), z(z_val) {}

    vector3 operator+(const vector3& other) const;

    vector3 operator*(type scalar) const;

    vector3& operator+=(const vector3& other);

	vector3& operator*=(type scalar);

    vector3& operator/=(type scalar);

    bool isValid() const;

    void empty();
};