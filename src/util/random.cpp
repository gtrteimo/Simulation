#include "util/random.hpp"

static std::mt19937 rng(std::random_device{}());
static std::uniform_real_distribution<type> dist0_1(0.0f, 1.0f);
static std::uniform_int_distribution<unsigned int> dist0_255(0, 255);

type randPer() {
    return dist0_1(rng);
}
vector2 randPer2() {
    return {randPer(), randPer()};
}
vector3 randPer3() {
    return {randPer(), randPer(), randPer()};
}

uint8_t randByte() {
    return static_cast<uint8_t>(dist0_255(rng));
}
colourRGB randColour(){
    return {randByte(), randByte(), randByte()};
}
colourRGB randLightColour(){
    uint8_t r = randByte();
    uint8_t g = randByte();
    uint8_t b = 0;
    if (510 - (r+g) > 255) {
        b = randByte();
    } else {
        b = static_cast<uint8_t>(510 - static_cast<uint16_t>(r+g));
    }
    return {r, g, b};
}

