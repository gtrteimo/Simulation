#include <variant>

#include "util/vector2.hpp" 

struct Mouse {
    struct LeftClick {
        vector2 position;
    };
    struct RightClick {
        vector2 position;
    };

    struct MiddleClick {
       vector2 position;
    };
};

struct Keyboard {
    struct KeyPressed {
        char keyCode;
        vector2 position;
    };
    struct KeyReleased {
        char keyCode;
        vector2 position;
    };
    struct KeyHeld {
        char keyCode;
    };
};

using InputType = std::variant<
        Mouse::LeftClick,
        Mouse::RightClick,
        Mouse::MiddleClick,
        Keyboard::KeyPressed,
        Keyboard::KeyReleased,
        Keyboard::KeyHeld
    >;
