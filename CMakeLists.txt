cmake_minimum_required(VERSION 3.10.0)
project(raylib_test VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)


# Tell CMake to search /usr/local for installed packages
list(APPEND CMAKE_PREFIX_PATH "/home/mehemmed/Documents")
# Find raylib installed on the system (e.g., via sudo make install or apt)
find_package(raylib REQUIRED)

add_executable(raylib_test main.c ai.c)

# Link raylib and its system dependencies
target_link_libraries(raylib_test PRIVATE raylib)
