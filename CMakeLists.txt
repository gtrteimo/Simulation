cmake_minimum_required(VERSION 3.10)

set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

add_subdirectory(.dependencies/glfw-3.4/)

project(Simulation VERSION 0.1 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS src/*.cpp)
add_executable(${PROJECT_NAME} .dependencies/glad.c ${SRC_FILES})

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${CMAKE_SOURCE_DIR}/.dependencies  ${CMAKE_SOURCE_DIR}/src/include)

target_link_libraries(${CMAKE_PROJECT_NAME} glfw)

target_compile_options(${PROJECT_NAME} PRIVATE
    -pedantic-errors 
    -Wall 
    -Wextra 
    -Wconversion 
    -Wsign-conversion 
    -Werror
    $<$<CONFIG:Debug>:-g>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

set_source_files_properties(.dependencies/glad.c PROPERTIES COMPILE_FLAGS "-w")
