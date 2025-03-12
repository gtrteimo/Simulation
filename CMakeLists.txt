cmake_minimum_required(VERSION 3.10)
project(Simulation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include_directories(.dependencies)
add_executable(Simulation .dependencies/glad.c src/main.cpp src/gui/window.cpp)

target_compile_options(Simulation PRIVATE
    -std=c++20
    -fdiagnostics-color=always
    -pedantic-errors
    -Wall
    # -Weffc++
    -Wextra
    -Wconversion
    -Wsign-conversion
    -Werror
    # -lGL
    # -lGLU
    # -lglut
    -lm
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
)

find_package(glfw3 REQUIRED)

set_source_files_properties(.dependencies/glad.c PROPERTIES COMPILE_FLAGS "-w")

target_link_libraries(Simulation glfw GL)
