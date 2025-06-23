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
colourRGB randDarkColour(){
    uint8_t r = randByte() / 2;
    uint8_t g = randByte() / 2;
    uint8_t b = randByte() / 2;
    return {r, g, b};
}

colourRGB randLightColour() {
    uint8_t r = randByte() / 2 + 128;
    uint8_t g = randByte() / 2 + 128;
    uint8_t b = randByte() / 2 + 128;
    return {r, g, b};
}

colourRGB randSingleColour() {
    uint8_t primary_component = randByte() / 3 + 170;
    
    uint8_t secondary_low = randByte() / 3;
    uint8_t secondary_mid = randByte() / 2 + 64;


    switch (randByte() % 10) {
        case 0: return {primary_component, secondary_low, secondary_low};
        case 1: return {secondary_low, primary_component, secondary_low};
        case 2: return {secondary_low, secondary_low, primary_component};
        
        case 3: return {primary_component, primary_component, secondary_low};
        case 4: return {primary_component, secondary_low, primary_component};
        case 5: return {secondary_low, primary_component, primary_component};

        case 6: return {primary_component, secondary_mid, secondary_low};
        case 7: return {primary_component, secondary_low, secondary_mid};
        case 8: return {secondary_mid, primary_component, secondary_low};
        case 9: return {secondary_low, secondary_mid, primary_component};
        
        default: return {255, 255, 255}; //Not possible but still
    }
}


