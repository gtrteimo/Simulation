cmake_minimum_required(VERSION 3.18) # Keep 3.18+ for better CUDA features

# --- Policy to silence FindCUDA warning ---
cmake_policy(SET CMP0146 NEW)

# --- Project Definition ---
project(Simulation VERSION 0.1 LANGUAGES C CXX CUDA)

# --- Find CUDA Toolkit ---
find_package(CUDAToolkit REQUIRED)
message(STATUS "Found CUDA toolkit.")
message(STATUS "  CUDA compiler ID: ${CMAKE_CUDA_COMPILER_ID}")

# --- Project Options ---
option(SIM_USE_FETCHCONTENT_FOR_GLFW "Download GLFW using FetchContent instead of using a local copy" ON)
option(SIM_GLFW_STATIC_LIB "Build GLFW as a static library" ON)

# Option for CUDA architecture(s)
# Note: Build log indicates compilation for 52 despite this setting.
# Ensure CMake cache is cleared if you intend to target 80;86;89.
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures to build for (e.g., 75;86 or native)")
endif()
message(STATUS "  Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}") # Use this to verify the setting

# --- C++ Standard ---
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set CUDA standard too
set(CMAKE_CUDA_STANDARD 17) # Uncommented and set
set(CMAKE_CUDA_STANDARD_REQUIRED True)

# --- Dependencies (GLFW) ---
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
    FetchContent_Declare(glfw GIT_REPOSITORY https://github.com/glfw/glfw.git GIT_TAG 3.4)
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
        set(GLFW_BUILD_SHARED OFF CACHE BOOL "Build GLFW as a static library" FORCE)
    else()
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries globally" FORCE)
        set(GLFW_BUILD_SHARED ON CACHE BOOL "Build GLFW as a shared library" FORCE)
    endif()
    add_subdirectory(.dependencies/glfw-3.4)
endif()

# --- Source Files ---
file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS
    "src/*.cpp"
    "src/*.c"
    "src/*.cu"
)

# Add glad.c
set(GLAD_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glad.c")
if(NOT EXISTS "${GLAD_SOURCE}")
    message(FATAL_ERROR "glad.c not found at ${GLAD_SOURCE}. Please generate and place it there.")
endif()

# --- Create Executable ---
add_executable(${PROJECT_NAME}
    ${GLAD_SOURCE}
    ${SRC_FILES}
)

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src/"
    "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies" # For glad.h
)

# --- Link Libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

# --- Compile Options ---
# Configuration-specific options (Debug/Release)
target_compile_options(${PROJECT_NAME} PRIVATE
  $<$<CONFIG:Debug>:-g> # Debug symbols (applies to C, CXX, CUDA host)
  $<$<CONFIG:Release>:-O3> # Optimization (applies to C, CXX, CUDA host)
)

# --- IMPORTANT CHANGE 1: Disable host flag propagation ---
# This prevents duplicate and malformed host compiler flags.
# CMake and the VS generator will handle default host flags correctly.
set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF) # Changed from ON

# --- IMPORTANT CHANGE 2: Suppress deprecated GPU targets warning correctly ---
# Add this flag specifically for the CUDA compilation step.
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>
)

# --- Source File Specific Properties ---
# Suppress warnings for the glad.c file
if(MSVC)
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "/w")
else()
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "-w")
endif()

# --- Output Directories (Optional) ---
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
if(CMAKE_CONFIGURATION_TYPES)
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${EXECUTABLE_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
    endforeach()
else()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
endif()

# --- Installation (Optional) ---
# ...

# --- Packaging with CPack (Optional) ---
# ...

message(STATUS "CMake configuration finished for ${PROJECT_NAME}.")
message(STATUS "  Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "  Binary directory: ${CMAKE_BINARY_DIR}")
if(CMAKE_BUILD_TYPE)
    message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
else()
    message(STATUS "  Build type: Multi-config generator (e.g., Visual Studio)")
endif()
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "  CUDA Compiler: ${CMAKE_CUDA_COMPILER_ID} (${CMAKE_CUDA_COMPILER})")
message(STATUS "  Using C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Using CUDA Standard: ${CMAKE_CUDA_STANDARD}") # Added message
message(STATUS "  Building for CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")