cmake_minimum_required(VERSION 3.18) # Keep 3.18+ for better CUDA features

# --- Policy to silence FindCUDA warning ---
# CMP0146 indicates that FindCUDA is deprecated.
# Setting this policy to NEW tells CMake you're aware and proceeding with newer practices.
cmake_policy(SET CMP0146 NEW)

# --- Project Definition ---
# Add CUDA as a required language - this is still the modern way to enable .cu compilation
project(Simulation VERSION 0.1 LANGUAGES C CXX CUDA)

# --- Find CUDA Toolkit (Modern Way) ---
# Use the modern package name 'CUDAToolkit' instead of the old 'CUDA'
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA toolkit.")
# The modern find_package doesn't typically expose CUDA_TOOLKIT_ROOT_DIR directly
# message(STATUS "  CUDA toolkit root not directly exposed by modern find_package(CUDAToolkit)")
message(STATUS "  CUDA compiler ID: ${CMAKE_CUDA_COMPILER_ID}") # This is set by the project command


# --- Project Options (can be changed by user e.g. with cmake-gui or -D) ---
option(SIM_USE_FETCHCONTENT_FOR_GLFW "Download GLFW using FetchContent instead of using a local copy" ON)
option(SIM_GLFW_STATIC_LIB "Build GLFW as a static library" ON) # Default to static for GLFW

# Option for CUDA architecture(s)
# You should set this based on your target GPU(s).
# This property is automatically handled by the CUDA language support and find_package(CUDAToolkit).
# The 'native' option attempts to detect the architecture of the build machine's GPU.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (e.g., 75;86 or native)")
endif()
message(STATUS "Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")


# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# You can optionally set the CUDA standard too (requires CMake 3.18+)
# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED True)


# --- Dependencies (GLFW) ---
# (This section remains the same as in your previous working CMakeLists.txt)

if(SIM_USE_FETCHCONTENT_FOR_GLFW)
    include(FetchContent)
    message(STATUS "Using FetchContent to manage GLFW.")

    set(GLFW_BUILD_EXAMPLES OFF)
    set(GLFW_BUILD_TESTS OFF)
    set(GLFW_BUILD_DOCS OFF)
    set(GLFW_INSTALL OFF)
    if(SIM_GLFW_STATIC_LIB)
        set(BUILD_SHARED_LIBS OFF)
        set(GLFW_BUILD_SHARED OFF)
    else()
        set(BUILD_SHARED_LIBS ON)
        set(GLFW_BUILD_SHARED ON)
    endif()

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
    )
    FetchContent_MakeAvailable(glfw)

else()
    message(STATUS "Using local GLFW from .dependencies/glfw-3.4.")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glfw-3.4/CMakeLists.txt")
        message(FATAL_ERROR "GLFW source not found in .dependencies/glfw-3.4/. "
                            "Please clone/place it there or enable SIM_USE_FETCHCONTENT_FOR_GLFW.")
    endif()

    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "Install GLFW" FORCE)
    if(SIM_GLFW_STATIC_LIB)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries globally" FORCE)
        set(GLFW_BUILD_SHARED OFF CACHE BOOL "Build GLFW as a shared library" FORCE)
    else()
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries globally" FORCE)
        set(GLFW_BUILD_SHARED ON CACHE BOOL "Build GLFW as a shared library" FORCE)
    endif()

    add_subdirectory(.dependencies/glfw-3.4)
endif()


# --- Source Files ---
# Include .cu files in the glob
file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.c"
    "src/*.cu"  # Include CUDA source files
)

# Add glad.c
set(GLAD_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glad.c")
if(NOT EXISTS "${GLAD_SOURCE}")
    message(FATAL_ERROR "glad.c not found at ${GLAD_SOURCE}. Please generate and place it there.")
endif()

# --- Create Executable ---
# Use standard add_executable. CMake will use nvcc for .cu files
add_executable(${PROJECT_NAME}
    ${GLAD_SOURCE}
    ${SRC_FILES}
)

# --- Target-specific C++/CUDA Standard (alternative) ---
# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)
# target_compile_features(${PROJECT_NAME} PRIVATE cuda_std_17) # Requires CMake 3.18+

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies" # For glad.h if it's directly in .dependencies
    # If glad headers are in a subfolder, e.g., .dependencies/glad/include:
    # "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glad/include"
)
# GLFW include directories are handled automatically by target_link_libraries(glfw)


# --- Link Libraries ---
# Use keyword signature for ALL links.
# Link against GLFW
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# Link against the CUDA Toolkit (includes cudart). This uses the modern interface target.
# Place this after linking other libraries that might depend on CUDA runtime symbols if order matters.
target_link_libraries(${PROJECT_NAME} PRIVATE CUDAToolkit::Toolkit)


# --- Compile Options ---
# Common options for C++/C/CUDA host code
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
)

# Common warning flags (most compilers understand these or ignore them)
# Example:
# if(NOT MSVC)
#     target_compile_options(${PROJECT_NAME} PRIVATE -Wextra -pedantic)
# endif()


# Configuration-specific options
# Example:
# target_compile_options(${PROJECT_NAME} PRIVATE
#   $<$<CONFIG:Debug>:-g> # Debug symbols for Debug configuration
#   $<$<CONFIG:Release>:-O3> # Optimization level 3 for Release
# )


# Compiler-specific and OS-specific warning flags
# Example:
# if(MSVC)
#   target_compile_options(${PROJECT_NAME} PRIVATE /W4) # Warning level 4 for MSVC
# elseif(GNU OR Clang)
#   target_compile_options(${PROJECT_NAME} PRIVATE -Wshadow -Wformat=2) # Add more warnings for GCC/Clang
# endif()


# --- CUDA Specific Compile Options ---
# These options are passed specifically to the NVCC compiler.
# CMAKE_CUDA_ARCHITECTURES property handles the architecture flags (-gencode/-arch).
# Add other specific NVCC flags here if needed.
# target_compile_options(${PROJECT_NAME} PRIVATE
#    --compiler-options "-Wno-unused-parameter" # Example: pass option to host compiler
#    -Wno-deprecated-gpu-targets # Example: disable NVCC warning
# )


# Ensure host compiler flags are propagated to NVCC's host pass (usually ON by default, but explicit is clear)
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS ON)


# --- Source File Specific Properties ---
# Silence warnings for glad.c
# Note: This overrides other compile options for this specific file.
if(MSVC)
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "-w")
endif()


# --- Output Directories (Optional) ---
# Set output directories based on build configuration
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (like Visual Studio)
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${EXECUTABLE_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
    endforeach()
else() # Single-config generator (like Makefile, Ninja)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
endif()


# --- Installation (Optional) ---
# install(TARGETS ${PROJECT_NAME}
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib
# )

# --- Packaging with CPack (Optional) ---
# include(CPack)


message(STATUS "CMake configuration finished for ${PROJECT_NAME}.")
message(STATUS "  Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "  Binary directory: ${CMAKE_BINARY_DIR}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE} (if single-config generator)")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} (${CMAKE_CUDA_COMPILER})")
message(STATUS "  Using C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")