cmake_minimum_required(VERSION 3.16)
project(kalman_filters LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies. Eigen is header-only; a vendored copy under third_party/eigen
# is used when the system does not provide one.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    set(KALMAN_VENDORED_EIGEN "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
    if(EXISTS "${KALMAN_VENDORED_EIGEN}/Eigen/Core")
        message(STATUS "Using vendored Eigen at ${KALMAN_VENDORED_EIGEN}")
        add_library(kalman_eigen INTERFACE)
        # SYSTEM so Eigen's own -Wconversion/-Wsign-conversion noise does not
        # drown out this project's diagnostics.
        target_include_directories(kalman_eigen SYSTEM INTERFACE
            "${KALMAN_VENDORED_EIGEN}")
        add_library(Eigen3::Eigen ALIAS kalman_eigen)
    else()
        message(FATAL_ERROR
            "Eigen3 not found and no vendored copy at third_party/eigen. "
            "Install Eigen (e.g. 'brew install eigen') or unpack Eigen 3.4 into "
            "third_party/eigen.")
    endif()
else()
    message(STATUS "Using system Eigen ${Eigen3_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Warnings: Google style is a formatting contract, but keep the code honest
# with a strict warning set (and -Werror in CI so warnings cannot accumulate).
# ---------------------------------------------------------------------------
option(KALMAN_WERROR "Treat warnings as errors" OFF)

set(KALMAN_WARNINGS
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
    -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
    -Wdouble-promotion -Wformat=2)
if(KALMAN_WERROR)
    list(APPEND KALMAN_WARNINGS -Werror)
endif()

# ---------------------------------------------------------------------------
# Library
# ---------------------------------------------------------------------------
add_library(kalman
    src/extended_kalman_filter.cpp
    src/numerics.cpp
    src/unscented_kalman_filter.cpp)
add_library(kalman::kalman ALIAS kalman)

target_include_directories(kalman
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kalman PUBLIC Eigen3::Eigen)
target_compile_options(kalman PRIVATE ${KALMAN_WARNINGS})

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
add_executable(linear_tracking_example examples/linear_tracking_example.cpp)
target_link_libraries(linear_tracking_example PRIVATE kalman::kalman)
target_compile_options(linear_tracking_example PRIVATE ${KALMAN_WARNINGS})

add_executable(pendulum_ekf_example examples/pendulum_ekf_example.cpp)
target_link_libraries(pendulum_ekf_example PRIVATE kalman::kalman)
target_compile_options(pendulum_ekf_example PRIVATE ${KALMAN_WARNINGS})

add_executable(imu_attitude_example examples/imu_attitude_example.cpp)
target_link_libraries(imu_attitude_example PRIVATE kalman::kalman)
target_compile_options(imu_attitude_example PRIVATE ${KALMAN_WARNINGS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_library(kalman_test_support STATIC tests/testing.cpp)
target_include_directories(kalman_test_support
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(kalman_test_support PUBLIC kalman::kalman)
target_compile_options(kalman_test_support PRIVATE ${KALMAN_WARNINGS})

add_executable(kalman_tests
    tests/attitude_estimation_test.cpp
    tests/extended_kalman_filter_test.cpp
    tests/linear_kalman_filter_test.cpp
    tests/numerics_test.cpp
    tests/quaternion_test.cpp
    tests/unscented_kalman_filter_test.cpp)
target_link_libraries(kalman_tests PRIVATE kalman_test_support)
target_compile_options(kalman_tests PRIVATE ${KALMAN_WARNINGS})

add_test(NAME kalman_tests COMMAND kalman_tests)
