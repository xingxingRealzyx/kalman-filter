// Copyright 2024 The Kalman Filter Library Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Example: linear Kalman filter on a 2D constant-velocity track.
//
// The target is a car driving in a straight line at constant speed, and the
// only sensor is a GPS-like position measurement with 2 m (1 sigma) noise. The
// filter estimates [px, py, vx, vy] and, as importantly, reports how much it
// trusts that estimate in the covariance.
//
// Build (from the repository root):
//   cmake -S kalman-filter -B kalman-filter/build
//   cmake --build kalman-filter/build -j
//   ./kalman-filter/build/linear_tracking_example

#include <iomanip>
#include <iostream>
#include <random>

#include "kalman/linear_kalman_filter.h"
#include "kalman/numerics.h"

namespace {

// Continuous white-noise acceleration model: sampling an acceleration that is
// constant over each interval with variance sigma_a^2 gives this discrete Q.
Eigen::Matrix4d BuildProcessNoise(double dt, double acceleration_variance) {
    Eigen::Matrix4d q = Eigen::Matrix4d::Zero();
    const double dt2 = dt * dt;
    const double dt3 = dt2 * dt;
    const double dt4 = dt3 * dt;
    for (int axis = 0; axis < 2; ++axis) {
        const int position = axis;
        const int velocity = axis + 2;
        q(position, position) = acceleration_variance * dt4 / 4.0;
        q(position, velocity) = acceleration_variance * dt3 / 2.0;
        q(velocity, position) = q(position, velocity);
        q(velocity, velocity) = acceleration_variance * dt2;
    }
    return q;
}

}  // namespace

int main() {
    constexpr double kDt = 0.1;                 // 10 Hz GPS
    constexpr double kPositionSigma = 2.0;      // [m]
    constexpr double kAccelerationSigma = 0.5;  // [m/s^2] unmodelled manoeuvres

    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    a(0, 2) = kDt;
    a(1, 3) = kDt;
    const Eigen::Matrix4d q =
        BuildProcessNoise(kDt, kAccelerationSigma * kAccelerationSigma);

    Eigen::Matrix<double, 2, 4> c = Eigen::Matrix<double, 2, 4>::Zero();
    c(0, 0) = 1.0;
    c(1, 1) = 1.0;
    const Eigen::Matrix2d r =
        Eigen::Matrix2d::Identity() * kPositionSigma * kPositionSigma;

    // Ground truth: starts 40 m east, 30 m north, drives north-east at 8 m/s.
    Eigen::Vector4d truth;
    truth << 40.0, 30.0, 5.66, 5.66;

    // Initial guess: 10 m off in each position, velocity unknown (0), with a
    // deliberately loose covariance to say "I do not know yet".
    Eigen::Vector4d initial;
    initial << truth(0) + 10.0, truth(1) - 10.0, 0.0, 0.0;
    Eigen::Matrix4d initial_covariance = Eigen::Matrix4d::Zero();
    initial_covariance.diagonal() << 25.0, 25.0, 25.0, 25.0;

    kalman::LinearKalmanFilter<4, 2> filter(initial, initial_covariance);

    std::mt19937 generator(2024);
    std::normal_distribution<double> position_noise(0.0, kPositionSigma);

    std::cout
        << "step  time[s]   measured            estimated                  "
           "1-sigma [m]   pos error [m]\n";
    std::cout << std::fixed;

    constexpr int kSteps = 60;
    for (int step = 0; step < kSteps; ++step) {
        truth = a * truth;
        Eigen::Vector2d measurement;
        measurement << truth(0) + position_noise(generator),
            truth(1) + position_noise(generator);

        filter.Predict(a, q);
        if (!filter.Update(c, measurement, r)) {
            std::cerr << "update rejected at step " << step << "\n";
            return 1;
        }

        const Eigen::Vector4d& estimate = filter.state();
        const Eigen::Matrix4d& covariance = filter.covariance();
        const double position_error =
            (estimate.head<2>() - truth.head<2>()).norm();

        if (step % 5 == 0 || step == kSteps - 1) {
            std::cout << std::setw(4) << step << std::setw(8)
                      << std::setprecision(1) << (step + 1) * kDt << "   ("
                      << std::setprecision(2) << std::setw(7) << measurement(0)
                      << ", " << std::setw(7) << measurement(1) << ")   ("
                      << std::setw(7) << estimate(0) << ", " << std::setw(7)
                      << estimate(1) << ", " << std::setw(6) << estimate(2)
                      << ", " << std::setw(6) << estimate(3) << ")   "
                      << std::setw(8) << std::sqrt(covariance(0, 0)) << "   "
                      << std::setw(10) << position_error << "\n";
        }
    }

    const Eigen::Vector4d error = filter.state() - truth;
    std::cout << "\ntrue velocity      [" << truth(2) << ", " << truth(3)
              << "] m/s\n"
              << "estimated velocity [" << filter.state()(2) << ", "
              << filter.state()(3) << "] m/s\n"
              << "position error     " << error.head<2>().norm() << " m\n"
              << "velocity error     " << error.tail<2>().norm() << " m/s\n"
              << "1-sigma position   " << std::sqrt(filter.covariance()(0, 0))
              << " m\n"
              << "1-sigma velocity   " << std::sqrt(filter.covariance()(2, 2))
              << " m/s\n";
    return 0;
}
