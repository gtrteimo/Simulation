#pragma once

#include "util/types.h"

struct vector3 {
    type x;
	type y;
	type z;

    vector3 operator+(const vector3& other) const;

    vector3 operator*(type scalar) const;

    vector3& operator+=(const vector3& other);

	vector3& operator*=(type scalar);
};