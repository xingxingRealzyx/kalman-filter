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

#include "kalman/unscented_kalman_filter.h"

#include <cmath>
#include <limits>
#include <random>

#include "kalman/extended_kalman_filter.h"
#include "kalman/linear_kalman_filter.h"
#include "kalman/numerics.h"
#include "tests/testing.h"

namespace {

constexpr double kGravityOverLength = 10.0;
constexpr double kDamping = 0.2;
constexpr double kSampleTime = 0.05;
constexpr double kAngleSigma = 0.5 * M_PI / 180.0;
constexpr int kSubSteps = 20;

Eigen::Vector2d PendulumDynamics(const Eigen::Vector2d& state) {
    Eigen::Vector2d derivative;
    derivative << state(1),
        -kGravityOverLength * std::sin(state(0)) - kDamping * state(1);
    return derivative;
}

Eigen::Vector2d PendulumStep(const Eigen::Vector2d& state) {
    const double h = kSampleTime / kSubSteps;
    Eigen::Vector2d x = state;
    for (int i = 0; i < kSubSteps; ++i) {
        const Eigen::Vector2d k1 = PendulumDynamics(x);
        const Eigen::Vector2d k2 = PendulumDynamics(x + 0.5 * h * k1);
        const Eigen::Vector2d k3 = PendulumDynamics(x + 0.5 * h * k2);
        const Eigen::Vector2d k4 = PendulumDynamics(x + h * k3);
        x += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    kalman::NormalizeAnglesInPlace(x.head<1>());
    return x;
}

Eigen::VectorXd PendulumTransition(const Eigen::VectorXd& state) {
    return PendulumStep(state);
}

Eigen::VectorXd PendulumMeasurement(const Eigen::VectorXd& state) {
    Eigen::VectorXd measurement(1);
    measurement << state(0);
    return measurement;
}

Eigen::VectorXd BearingRangeMeasurement(const Eigen::VectorXd& state) {
    Eigen::VectorXd measurement(2);
    measurement << std::hypot(state(0), state(1)),
        std::atan2(state(1), state(0));
    return measurement;
}

Eigen::VectorXd WrapAngle(const Eigen::VectorXd& state) {
    Eigen::VectorXd wrapped = state;
    kalman::NormalizeAnglesInPlace(wrapped.head<1>());
    return wrapped;
}

}  // namespace

TEST(UnscentedKalmanFilter, MatchesLinearKalmanFilterOnLinearSystem) {
    // The unscented transform is exact for linear maps (up to floating point),
    // so on a linear model the UKF and the linear Kalman filter must produce
    // the same estimate. Both the matrix overloads and the lambda overloads are
    // checked.
    constexpr double kDt = 0.1;
    Eigen::Matrix2d a;
    a << 1.0, kDt, 0.0, 1.0;
    Eigen::Matrix2d q = Eigen::Matrix2d::Identity() * 1e-3;
    Eigen::Matrix<double, 1, 2> c;
    c << 1.0, 0.0;
    Eigen::Matrix<double, 1, 1> r;
    r << 0.25;

    Eigen::Vector2d initial;
    initial << 4.0, -1.0;
    const Eigen::Matrix2d initial_covariance =
        (Eigen::Vector2d() << 2.0, 0.5).finished().asDiagonal();

    kalman::LinearKalmanFilter<2, 1> reference(initial, initial_covariance);
    kalman::UnscentedKalmanFilter ukf(initial, initial_covariance);
    kalman::UnscentedKalmanFilter lambda_ukf(initial, initial_covariance);

    std::mt19937 generator(20240517);
    std::normal_distribution<double> noise(0.0, 0.5);
    for (int step = 0; step < 60; ++step) {
        const double value = 3.0 + noise(generator);
        reference.Predict(a, q);
        CHECK_TRUE(ukf.PredictLinear(a, q));
        CHECK_TRUE(lambda_ukf.Predict(
            kalman::UnscentedKalmanFilter::StateTransitionFunction(
                [&a](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                    return a * x;
                }),
            q));
        CHECK_TRUE(reference.Update(c, Eigen::VectorXd::Constant(1, value), r));
        CHECK_TRUE(
            ukf.CorrectLinear(c, Eigen::VectorXd::Constant(1, value), r));
        CHECK_TRUE(lambda_ukf.Correct(
            kalman::UnscentedKalmanFilter::MeasurementFunction(
                [&c](const Eigen::VectorXd& x) -> Eigen::VectorXd {
                    return c * x;
                }),
            Eigen::VectorXd::Constant(1, value), r));

        CHECK_NEAR((reference.state() - ukf.state()).norm(), 0.0, 1e-9);
        CHECK_NEAR((reference.covariance() - ukf.covariance()).norm(), 0.0,
                   1e-9);
        CHECK_NEAR((reference.state() - lambda_ukf.state()).norm(), 0.0, 1e-9);
        CHECK_NEAR((reference.covariance() - lambda_ukf.covariance()).norm(),
                   0.0, 1e-9);
    }
}

TEST(UnscentedKalmanFilter, UnscentedTransformReproducesMeanAndCovariance) {
    // Passing the sigma points through the identity must return the input mean
    // and covariance exactly: this pins down the weights and the sigma point
    // construction, which is where UKF bugs live.
    Eigen::Vector3d mean;
    mean << 1.0, -2.0, 0.5;
    Eigen::Matrix3d covariance;
    covariance << 2.0, 0.3, 0.0,  //
        0.3, 1.0, -0.2,           //
        0.0, -0.2, 0.5;

    kalman::UnscentedKalmanFilter filter(mean, covariance);
    const kalman::UnscentedKalmanFilter::StateTransitionFunction identity =
        [](const Eigen::VectorXd& x) -> Eigen::VectorXd { return x; };
    CHECK_TRUE(filter.Predict(identity, Eigen::MatrixXd::Zero(3, 3)));
    CHECK_NEAR((filter.state() - mean).norm(), 0.0, 1e-9);
    CHECK_NEAR((filter.covariance() - covariance).norm(), 0.0, 1e-9);

    // A rotation is still a linear map: the transform must agree with the
    // analytic result to the same precision.
    Eigen::Matrix3d rotation;
    rotation << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    kalman::UnscentedKalmanFilter rotated(mean, covariance);
    CHECK_TRUE(rotated.PredictLinear(rotation, Eigen::MatrixXd::Zero(3, 3)));
    CHECK_NEAR((rotated.state() - rotation * mean).norm(), 0.0, 1e-9);
    CHECK_NEAR(
        (rotated.covariance() - rotation * covariance * rotation.transpose())
            .norm(),
        0.0, 1e-9);
}

TEST(UnscentedKalmanFilter, TracksPendulumWithoutJacobians) {
    constexpr int kSteps = 400;
    constexpr int kSettlingSteps = 150;

    std::mt19937 generator(20240517);
    std::normal_distribution<double> measurement_noise(0.0, kAngleSigma);
    std::normal_distribution<double> process_noise(0.0, 1.0);

    Eigen::VectorXd initial(2);
    initial << 30.0 * M_PI / 180.0, 0.0;
    Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Zero(2, 2);
    initial_covariance.diagonal() << 1e-6, 1e-4;

    Eigen::MatrixXd process_noise_covariance = Eigen::MatrixXd::Zero(2, 2);
    process_noise_covariance.diagonal() << 1e-6, 1e-4;
    Eigen::MatrixXd measurement_noise_covariance(1, 1);
    measurement_noise_covariance << kAngleSigma * kAngleSigma;

    kalman::UnscentedKalmanFilter filter(initial, initial_covariance);
    Eigen::Vector2d truth;
    truth << 30.0 * M_PI / 180.0, 0.0;

    double angle_square_sum = 0.0;
    double rate_square_sum = 0.0;
    int counted = 0;
    for (int step = 0; step < kSteps; ++step) {
        truth = PendulumStep(truth);
        truth(0) += 1e-3 * process_noise(generator);
        truth(1) += 1e-2 * process_noise(generator);

        Eigen::VectorXd measurement(1);
        measurement << truth(0) + measurement_noise(generator);

        CHECK_TRUE(
            filter.Predict(PendulumTransition, process_noise_covariance));
        CHECK_TRUE(filter.Correct(PendulumMeasurement, measurement,
                                  measurement_noise_covariance));
        CHECK_TRUE(filter.covariance().allFinite());
        CHECK_TRUE(kalman::IsPositiveSemidefinite(filter.covariance()));

        if (step >= kSettlingSteps) {
            const double angle_error = kalman::AngularDifference(
                filter.state()(0), static_cast<double>(truth(0)));
            const double rate_error = filter.state()(1) - truth(1);
            angle_square_sum += angle_error * angle_error;
            rate_square_sum += rate_error * rate_error;
            ++counted;
        }
    }

    const double angle_rmse_deg =
        std::sqrt(angle_square_sum / counted) * 180.0 / M_PI;
    const double rate_rmse = std::sqrt(rate_square_sum / counted);
    CHECK_TRUE(angle_rmse_deg < 0.5);
    CHECK_TRUE(rate_rmse < 0.05);
    // The angle is observable and heavily measured: its variance must be small.
    CHECK_TRUE(filter.covariance()(0, 0) < 2.0 * kAngleSigma * kAngleSigma);
}

TEST(UnscentedKalmanFilter, HandlesBearingRangeTracking) {
    // Nonlinear measurement, linear dynamics. On the linear part the UKF is
    // exact; on the polar measurement it must still produce a consistent
    // estimate and a positive definite covariance.
    constexpr double kDt = 0.1;
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    a(0, 2) = kDt;
    a(1, 3) = kDt;

    Eigen::Matrix4d q = Eigen::Matrix4d::Zero();
    const double acceleration_variance = 0.0025;
    for (int axis = 0; axis < 2; ++axis) {
        const int p = axis;
        const int v = axis + 2;
        q(p, p) = acceleration_variance * std::pow(kDt, 4) / 4.0;
        q(p, v) = acceleration_variance * std::pow(kDt, 3) / 2.0;
        q(v, p) = q(p, v);
        q(v, v) = acceleration_variance * kDt * kDt;
    }

    Eigen::Matrix2d r = Eigen::Matrix2d::Zero();
    const double range_sigma = 3.0;
    const double bearing_sigma = 0.5 * M_PI / 180.0;
    r(0, 0) = range_sigma * range_sigma;
    r(1, 1) = bearing_sigma * bearing_sigma;

    Eigen::VectorXd initial(4);
    initial << 80.0, 120.0, 1.5, -2.0;
    Eigen::Vector4d truth;
    truth << 100.0, 100.0, 1.5, -2.0;
    Eigen::Matrix4d initial_covariance = Eigen::Matrix4d::Zero();
    initial_covariance.diagonal() << 25.0, 25.0, 1.0, 1.0;

    kalman::UnscentedKalmanFilter filter(initial, initial_covariance);

    std::mt19937 generator(99);
    std::normal_distribution<double> range_noise(0.0, range_sigma);
    std::normal_distribution<double> bearing_noise(0.0, bearing_sigma);

    for (int step = 0; step < 300; ++step) {
        truth = a * truth;
        // Small unmodelled manoeuvre, same as the EKF test.
        if (step == 150) {
            truth(2) += 0.05;
        }

        Eigen::VectorXd measurement(2);
        measurement << std::hypot(truth(0), truth(1)) + range_noise(generator),
            std::atan2(truth(1), truth(0)) + bearing_noise(generator);

        CHECK_TRUE(filter.PredictLinear(a, q));
        CHECK_TRUE(filter.Correct(BearingRangeMeasurement, measurement, r));
        CHECK_TRUE(filter.covariance().allFinite());
        CHECK_TRUE(kalman::IsPositiveSemidefinite(filter.covariance()));
    }

    const Eigen::Vector2d position_error =
        filter.state().head<2>() - truth.head<2>();
    CHECK_TRUE(position_error.norm() < 10.0);
    const double mahalanobis =
        position_error.transpose() *
        filter.covariance().topLeftCorner<2, 2>().inverse() * position_error;
    CHECK_TRUE(mahalanobis < 100.0);
    // The target keeps moving: the filter must have learned its velocity.
    CHECK_NEAR(filter.state()(2), truth(2), 1.0);
    CHECK_NEAR(filter.state()(3), truth(3), 1.0);
}

TEST(UnscentedKalmanFilter, AveragesWrappedAnglesCorrectly) {
    // Two states straddling the +/-pi discontinuity. Without wrapping, the
    // unscented mean would land at 0 rad (the worst possible answer); with the
    // caller wrapping inside the model function it must land at +/-pi.
    Eigen::Vector2d mean;
    mean << M_PI - 0.2, 0.0;
    Eigen::Matrix2d covariance = Eigen::MatrixXd::Zero(2, 2);
    covariance.diagonal() << 4e-4, 1e-6;

    kalman::UnscentedKalmanFilter filter(mean, covariance);
    // Sigma points straddle the discontinuity; wrapping each propagated point
    // keeps the recovered mean on the correct side of it.
    CHECK_TRUE(filter.Predict(WrapAngle, Eigen::MatrixXd::Zero(2, 2)));
    CHECK_TRUE(std::abs(std::abs(filter.state()(0)) - M_PI) < 0.25);
    CHECK_TRUE(filter.state()(0) > 0.0);  // not flipped through zero
    CHECK_NEAR(filter.state()(1), 0.0, 1e-9);
}

TEST(UnscentedKalmanFilter, RejectsInvalidInputWithoutCorruptingState) {
    kalman::UnscentedKalmanFilter filter(Eigen::Vector2d::Zero(),
                                         Eigen::Matrix2d::Identity());
    const Eigen::VectorXd state_before = filter.state();
    const Eigen::MatrixXd covariance_before = filter.covariance();

    // Wrong-sized process noise.
    CHECK_FALSE(filter.PredictLinear(Eigen::MatrixXd::Identity(3, 3),
                                     Eigen::MatrixXd::Identity(3, 3)));
    // Negative-definite process noise.
    CHECK_FALSE(filter.PredictLinear(Eigen::MatrixXd::Identity(2, 2),
                                     -Eigen::MatrixXd::Identity(2, 2)));
    // Non-finite transition output.
    const kalman::UnscentedKalmanFilter::StateTransitionFunction broken =
        [](const Eigen::VectorXd&) -> Eigen::VectorXd {
        Eigen::VectorXd bad(2);
        bad << std::numeric_limits<double>::infinity(), 0.0;
        return bad;
    };
    CHECK_FALSE(filter.Predict(broken, Eigen::MatrixXd::Zero(2, 2)));
    // Wrong-sized measurement noise.
    CHECK_FALSE(filter.Correct(PendulumMeasurement, Eigen::VectorXd::Zero(1),
                               Eigen::MatrixXd::Identity(2, 2)));
    // Non-finite measurement.
    Eigen::VectorXd bad_measurement(1);
    bad_measurement << std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(filter.Correct(PendulumMeasurement, bad_measurement,
                               Eigen::MatrixXd::Identity(1, 1)));
    // Negative-definite measurement noise.
    Eigen::MatrixXd bad_noise(1, 1);
    bad_noise << -1.0;
    CHECK_FALSE(filter.Correct(PendulumMeasurement, Eigen::VectorXd::Zero(1),
                               bad_noise));

    CHECK_TRUE((filter.state() - state_before).norm() == 0.0);
    CHECK_TRUE((filter.covariance() - covariance_before).norm() == 0.0);

    // A valid step still works afterwards.
    CHECK_TRUE(filter.PredictLinear(Eigen::Matrix2d::Identity(),
                                    Eigen::Matrix2d::Zero()));
    CHECK_TRUE(filter.Correct(PendulumMeasurement,
                              Eigen::VectorXd::Constant(1, 0.25),
                              Eigen::MatrixXd::Identity(1, 1)));
    CHECK_TRUE(filter.state().allFinite());
}

TEST(UnscentedKalmanFilter, ResetValidatesCovariance) {
    kalman::UnscentedKalmanFilter filter;
    CHECK_TRUE(
        filter.Reset(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()));
    Eigen::Matrix2d indefinite;
    indefinite << 1.0, 0.0, 0.0, -1.0;
    CHECK_FALSE(filter.Reset(Eigen::Vector2d::Zero(), indefinite));
    CHECK_NEAR((filter.covariance() - Eigen::Matrix2d::Identity()).norm(), 0.0,
               0.0);
}
