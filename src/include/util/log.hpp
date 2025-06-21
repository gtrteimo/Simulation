#include <iostream>
#include <type_traits>
#include <utility>

template<typename T, typename = void>
struct is_streamable : std::false_type {};

template<typename T>
struct is_streamable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>>
    : std::true_type {};

template<typename T>
T log(const T& value) {
    if constexpr (is_streamable<T>::value) {
        std::cout << value << std::endl;
    } else {
        std::cout << "[Unprintable Type]" << std::endl;
    }
    return value;
}
