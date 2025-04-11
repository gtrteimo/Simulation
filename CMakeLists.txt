cmake_minimum_required(VERSION 3.10)

add_subdirectory(.dependencies/glfw-3.4/)

project(Simulation)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${CMAKE_PROJECT_NAME} .dependencies/glad.c src/main.cpp src/gui/window.cpp)

# include_directories(${CMAKE_SOURCE_DIR}/.dependencies) #DEPRECATED
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/.dependencies)

target_link_libraries(${CMAKE_PROJECT_NAME} glfw GL) #DEPRECATED

target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE
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

# find_package(glfw3 REQUIRED) #DEPRECATED

set_source_files_properties(.dependencies/glad.c PROPERTIES COMPILE_FLAGS "-w")

# target_link_libraries(${CMAKE_PROJECT_NAME} glfw GL) #DEPRECATED
