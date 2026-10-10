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

# GCC reports false positives *inside Eigen's vectorized kernels*: for a fixed
# 1x1 Eigen matrix (8 bytes) its auto-vectorizer emits a 16-byte SIMD load/store
# and then diagnoses its own output as an out-of-bounds access, e.g.
#
#   aarch64: arm_neon.h ... "array subscript ... partly outside array bounds of
#            kalman::LinearKalmanFilter<2, 1>::MeasurementCovariance"
#   x86_64:  emmintrin.h ... the same diagnosis, via SSE
#
# Nothing in this repository can fix that, only suppress it, so the suppression
# is narrow (GCC only, these three warnings only) and is applied *after* the
# -Werror appended below, because -Werror converts them into errors. Clang does
# not produce them, and they are the only reason this project does not treat
# every warning as an error on every compiler.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND KALMAN_WARNINGS
        -Wno-array-bounds -Wno-stringop-overread -Wno-stringop-overflow)
endif()

if(KALMAN_WERROR)
    list(APPEND KALMAN_WARNINGS -Werror)
endif()

# ---------------------------------------------------------------------------
# Library: the filters and the math they need, and nothing else.
#   public API = include/kalman/*.h  (linear, extended, unscented, numerics)
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
# Optional module: attitude estimation (quaternion math + an IMU model).
#
# This lives in include/kalman/attitude/ and is NOT part of kalman::kalman:
# an IMU model is application code, and a user estimating something else should
# not have to compile it (or see it in the library's API). It is header-only, so
# consuming it costs one include; link kalman::attitude to get both.
# ---------------------------------------------------------------------------
add_library(kalman_attitude INTERFACE)
add_library(kalman::attitude ALIAS kalman_attitude)
target_include_directories(kalman_attitude
    INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kalman_attitude INTERFACE kalman::kalman)
target_compile_options(kalman_attitude INTERFACE ${KALMAN_WARNINGS})

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
target_link_libraries(imu_attitude_example PRIVATE kalman::attitude)
target_compile_options(imu_attitude_example PRIVATE ${KALMAN_WARNINGS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_library(kalman_test_support STATIC tests/testing.cpp)
target_include_directories(kalman_test_support
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(kalman_test_support PUBLIC kalman::attitude)
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
