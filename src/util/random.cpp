#include "util/random.hpp"

type randPer() {
    static std::mt19937 rng(std::random_device{}());
    static std::uniform_real_distribution<type> dist(0.0f, 1.0f);
    return dist(rng);
}
vector3 randPer3() {
    return {randPer(), randPer(), randPer()};
}
vector2 randPer2() {
    return {randPer(), randPer()};
}