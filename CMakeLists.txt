cmake_minimum_required(VERSION 3.12) # Increased for FetchContent, modern CMake features

project(Simulation VERSION 0.1 LANGUAGES C CXX)

# --- Project Options (can be changed by user e.g. with cmake-gui or -D) ---
option(SIM_USE_FETCHCONTENT_FOR_GLFW "Download GLFW using FetchContent instead of using a local copy" OFF)
option(SIM_GLFW_STATIC_LIB "Build GLFW as a static library" ON) # Default to static for GLFW

# --- C++ Standard ---
# Enforce C++17 or newer.
# This sets the CMAKE_CXX_STANDARD property on all targets created later.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF) # Prefer not to use compiler-specific extensions like -std=gnu++17

# --- Dependencies ---

# GLFW:
# Option 1: Using FetchContent (recommended for self-contained projects)
if(SIM_USE_FETCHCONTENT_FOR_GLFW)
    include(FetchContent)
    message(STATUS "Using FetchContent to manage GLFW.")

    # Set GLFW build options BEFORE FetchContent_MakeAvailable
    set(GLFW_BUILD_EXAMPLES OFF)
    set(GLFW_BUILD_TESTS OFF)
    set(GLFW_BUILD_DOCS OFF)
    set(GLFW_INSTALL OFF)
    if(SIM_GLFW_STATIC_LIB)
        set(BUILD_SHARED_LIBS OFF) # Affects GLFW build if it checks this
        set(GLFW_BUILD_SHARED OFF) # More specific to GLFW if it uses this var
    else()
        set(BUILD_SHARED_LIBS ON)
        set(GLFW_BUILD_SHARED ON)
    endif()

    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4 # Or latest stable tag, or specific commit
                          # Check https://github.com/glfw/glfw/releases for tags
        # For local development if you have GLFW source already:
        # SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glfw-3.4 # Uncomment if you prefer local source with FetchContent
    )
    FetchContent_MakeAvailable(glfw)

    # Restore BUILD_SHARED_LIBS if it was changed, so it doesn't affect other potential libraries
    # unless that's the desired global behavior. For an executable, this is less critical.
    # For this project, we assume we want our main executable to link statically to GLFW.
    # If you had other libraries in your project, you might want to reset BUILD_SHARED_LIBS here.

else()
    # Option 2: Using add_subdirectory (as in your original file)
    # Assumes GLFW source is in .dependencies/glfw-3.4
    message(STATUS "Using local GLFW from .dependencies/glfw-3.4.")
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glfw-3.4/CMakeLists.txt")
        message(FATAL_ERROR "GLFW source not found in .dependencies/glfw-3.4/. "
                            "Please clone/place it there or enable SIM_USE_FETCHCONTENT_FOR_GLFW.")
    endif()

    # Set GLFW build options BEFORE add_subdirectory
    # Using CACHE ... FORCE is strong; usually setting them as normal variables is enough
    # if the subproject checks them before defining its own options.
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "Build GLFW examples" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "Build GLFW tests" FORCE)
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "Build GLFW documentation" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "Install GLFW" FORCE) # Don't install GLFW when building our project
    if(SIM_GLFW_STATIC_LIB)
        # This tells GLFW (and potentially other subprojects) to build as static libs
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries globally" FORCE)
        # Some projects might use a specific variable like this:
        set(GLFW_BUILD_SHARED OFF CACHE BOOL "Build GLFW as a shared library" FORCE)
    else()
        set(BUILD_SHARED_LIBS ON CACHE BOOL "Build shared libraries globally" FORCE)
        set(GLFW_BUILD_SHARED ON CACHE BOOL "Build GLFW as a shared library" FORCE)
    endif()

    add_subdirectory(.dependencies/glfw-3.4)
endif()


# --- Source Files ---
# GLOB_RECURSE is convenient but be aware: if you add/remove files,
# you might need to re-run CMake manually (or touch CMakeLists.txt)
# unless CONFIGURE_DEPENDS is very reliable for your CMake version/generator.
# For more robust builds, especially in larger projects or CI, list files explicitly.
file(GLOB_RECURSE SRC_FILES
    CONFIGURE_DEPENDS # Ensures CMake re-runs if files matching the glob change
    "src/*.cpp"
    "src/*.c" # If you have any .c files in src/
)
# Example of explicit listing (more robust):
# set(SRC_FILES
#     src/main.cpp
#     src/engine/renderer.cpp
#     # ... other files
# )

# Add glad.c (assuming it's pre-generated and in .dependencies/)
# If glad.h is also there, its include path will be handled below.
set(GLAD_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glad.c")
if(NOT EXISTS "${GLAD_SOURCE}")
    message(FATAL_ERROR "glad.c not found at ${GLAD_SOURCE}. Please generate and place it there.")
endif()

add_executable(${PROJECT_NAME}
    ${GLAD_SOURCE}
    ${SRC_FILES}
)

# --- Target-specific C++ Standard (alternative to global set) ---
# If you prefer setting C++ standard per target:
# target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

# --- Include Directories ---
target_include_directories(${PROJECT_NAME} PUBLIC # Or PRIVATE if headers are not needed by users of this target (if it were a lib)
    "${CMAKE_CURRENT_SOURCE_DIR}/src/include"  # Your project's public headers
    "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies" # For glad.h (if it's directly in .dependencies)
    # If glad headers are in a subfolder, e.g., .dependencies/glad/include:
    # "${CMAKE_CURRENT_SOURCE_DIR}/.dependencies/glad/include"
)
# GLFW include directories are automatically added by `target_link_libraries`
# because a well-behaved CMake library (like GLFW) uses `target_include_directories`
# with INTERFACE or PUBLIC scope.

# --- Link Libraries ---
# PRIVATE: glfw is a build dependency, not propagated to link interface of ${PROJECT_NAME} (if it were a library)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

# --- Compile Options ---
# Common options for all compilers and configurations
target_compile_options(${PROJECT_NAME} PRIVATE
    # Add definitions that should apply to all builds, e.g. -DMY_MACRO
)

# Common warning flags (most compilers understand these or ignore them)
target_compile_options(${PROJECT_NAME} PRIVATE
    -Wall
    -pedantic-errors # -pedantic-errors is very strict, -pedantic is often a good balance
              # If you really want errors: -pedantic-errors
)

# Configuration-specific options
target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<CONFIG:Debug>:-g>                     # Debug: add debug symbols
    $<$<CONFIG:Release>:-O3 -DNDEBUG>         # Release: optimize, define NDEBUG
    $<$<CONFIG:RelWithDebInfo>:-O2 -g -DNDEBUG> # Release with Debug Info
    $<$<CONFIG:MinSizeRel>:-Os -DNDEBUG>      # Minimum Size Release
)

# Compiler-specific and OS-specific warning flags
if(MSVC) # Microsoft Visual C++
    target_compile_options(${PROJECT_NAME} PRIVATE
        /W4                # Warning level 4 (high)
        /WX                # Treat warnings as errors
        /permissive-       # Enforce standard conformance more strictly
        /wd4201            # Disable: nonstandard extension used: nameless struct/union (common in Windows headers)
        # Add other MSVC specific flags or /wdXXXX to disable specific warnings
    )
else() # GCC, Clang, and other non-MSVC compilers
    target_compile_options(${PROJECT_NAME} PRIVATE
        -Wextra            # Even more warnings
        -Werror            # Treat all enabled warnings as errors (can be aggressive)
        -Wconversion       # Warn on implicit conversions that may alter a value (can be noisy)
        -Wsign-conversion  # Warn for implicit sign conversions (can be noisy)
        # -Wshadow           # Warn when a local variable shadows another local variable, parameter or global
        -Wno-unused-parameter # Example: If you often have unused parameters you want to ignore
        -fPIC              # Required for shared libraries on some platforms, good practice for executables too
                           # if they might be part of position-independent code scenarios.
                           # For executables, it's often not strictly necessary but doesn't hurt.
    )

    # Specific to GCC
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
        target_compile_options(${PROJECT_NAME} PRIVATE
            # Add GCC-specific flags here if needed
            # e.g. -Wduplicated-branches
        )
    endif()

    # Specific to Clang
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(${PROJECT_NAME} PRIVATE
            # Add Clang-specific flags here if needed
            # e.g. -Wimplicit-fallthrough
        )
    endif()
endif()


# --- Source File Specific Properties ---
# Silence warnings for glad.c (which is generated C code)
# This is important as generated code might not adhere to your strict warning levels.
if(MSVC)
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "/w") # Turn off all warnings for glad.c with MSVC
else()
    set_source_files_properties(${GLAD_SOURCE} PROPERTIES COMPILE_FLAGS "-w") # Turn off all warnings for glad.c with GCC/Clang
endif()


# --- Output Directories (Optional but good practice for organization) ---
# This helps keep your build directory tidy, especially with multi-config generators (like VS)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${CMAKE_BUILD_TYPE}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${CMAKE_BUILD_TYPE}")

# For single-config generators, CMAKE_BUILD_TYPE might be empty in the configure step.
# A more robust way for output directories:
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/bin")
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}/lib")
if(CMAKE_CONFIGURATION_TYPES) # Multi-config generator (e.g. Visual Studio, Xcode)
    foreach(CONFIG ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER ${CONFIG} CONFIG_UPPER)
        set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${EXECUTABLE_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
        set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${CONFIG_UPPER} "${LIBRARY_OUTPUT_PATH}/${CONFIG}")
    endforeach()
else() # Single-config generator (e.g. Makefiles, Ninja)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${LIBRARY_OUTPUT_PATH}")
endif()


# --- Installation (Optional, if you plan to install your project) ---
# install(TARGETS ${PROJECT_NAME}
#     RUNTIME DESTINATION bin
#     LIBRARY DESTINATION lib
#     ARCHIVE DESTINATION lib/static
# )
# install(FILES src/include/myheader.h DESTINATION include) # Example

# --- Packaging with CPack (Optional) ---
# include(InstallRequiredSystemLibraries) # If you need to bundle system libs
# set(CPACK_GENERATOR "TGZ") # Or ZIP, RPM, DEB, NSIS etc.
# set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
# set(CPACK_PACKAGE_VERSION "${PROJECT_VERSION}")
# # ... other CPack settings
# include(CPack)

message(STATUS "CMake configuration finished for ${PROJECT_NAME}.")
message(STATUS "  Source directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "  Binary directory: ${CMAKE_BINARY_DIR}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE} (if single-config generator)")
message(STATUS "  C++ Compiler: ${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER})")
message(STATUS "  Using C++ Standard: ${CMAKE_CXX_STANDARD}")

