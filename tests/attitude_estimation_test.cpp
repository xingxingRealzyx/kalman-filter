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

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <cstdio>
#include <random>

#include "kalman/attitude/imu_attitude_model.h"
#include "kalman/attitude/quaternion.h"
#include "kalman/extended_kalman_filter.h"
#include "kalman/numerics.h"
#include "tests/testing.h"

namespace {

using kalman::imu::kQuaternionOffset;

constexpr double kDegrees = 180.0 / M_PI;

Eigen::VectorXd StateFromQuaternionForTest(const Eigen::Vector4d& quaternion) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
    state.segment<4>(kQuaternionOffset) = quaternion;
    return state;
}

Eigen::VectorXd MakeState(const Eigen::Vector3d& euler_rad) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
    state.segment<4>(kQuaternionOffset) =
        kalman::imu::QuaternionFromEulerZyx(euler_rad);
    return state;
}

}  // namespace

// ---------------------------------------------------------------------------
// Model level: the pieces that everything else rests on.
// ---------------------------------------------------------------------------
TEST(ImuAttitudeModel, QuaternionPropagationMatchesExactBodyRateRotation) {
    // The Taylor-series step matrix and the exact axis-angle propagator must
    // agree: they are two implementations of "rotate about the body axes by
    // |w| Ts". Any disagreement about the multiplication order shows up here
    // immediately (it is worth about a factor of two in the effective rate).
    std::mt19937 generator(11);
    std::normal_distribution<double> normal(0.0, 1.0);
    constexpr double kSampleTime = 0.01;
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d q =
            kalman::imu::QuaternionFromEulerZyx(Eigen::Vector3d(
                normal(generator), normal(generator), normal(generator)));
        Eigen::Vector3d rate;
        rate << 0.5 * normal(generator), 0.5 * normal(generator),
            0.5 * normal(generator);
        Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
        state.segment<4>(kQuaternionOffset) = q;
        const Eigen::Vector4d via_transition =
            kalman::imu::Transition(state, rate, kSampleTime)
                .segment<4>(kQuaternionOffset);
        const Eigen::Vector4d exact =
            kalman::imu::PropagateExact(q, rate, kSampleTime);
        CHECK_TRUE((via_transition - exact).norm() < 1e-12 ||
                   (via_transition + exact).norm() < 1e-12);
    }
}

TEST(ImuAttitudeModel, TransitionJacobianKeepsTangentVectorsTangent) {
    // The Jacobian must not rotate a tangent vector off the unit sphere: the
    // propagation renormalizes, so its derivative is a tangent-space map.
    std::mt19937 generator(23);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int trial = 0; trial < 10; ++trial) {
        const Eigen::Vector4d q =
            kalman::imu::QuaternionFromEulerZyx(Eigen::Vector3d(
                normal(generator), normal(generator), normal(generator)));
        const Eigen::VectorXd state = StateFromQuaternionForTest(q);
        const Eigen::Matrix4d jacobian = kalman::imu::TransitionJacobian(state);
        for (int probe = 0; probe < 5; ++probe) {
            Eigen::Vector4d tangent = Eigen::Vector4d::Random();
            const Eigen::Vector4d reference =
                state.segment<4>(kQuaternionOffset);
            tangent -=
                reference * (reference.dot(tangent) / reference.squaredNorm());
            const Eigen::Vector4d image = jacobian * tangent;
            CHECK_NEAR(reference.dot(image), 0.0, 1e-12);
        }
    }
}

TEST(ImuAttitudeModel, AccelerometerJacobianMatchesCentralDifferences) {
    std::mt19937 generator(31);
    std::normal_distribution<double> normal(0.0, 1.0);
    constexpr double kStep = 1e-6;
    for (int trial = 0; trial < 10; ++trial) {
        const Eigen::VectorXd state = MakeState(Eigen::Vector3d(
            normal(generator), normal(generator), normal(generator)));
        const Eigen::Matrix<double, 3, 4> analytic =
            kalman::imu::AccelerometerJacobian(state);
        Eigen::Matrix<double, 3, 4> numerical =
            Eigen::Matrix<double, 3, 4>::Zero();
        for (int column = 0; column < 4; ++column) {
            Eigen::VectorXd forward = state;
            Eigen::VectorXd backward = state;
            forward(column) += kStep;
            backward(column) -= kStep;
            numerical.col(column) =
                (kalman::imu::AccelerometerPrediction(forward) -
                 kalman::imu::AccelerometerPrediction(backward)) /
                (2.0 * kStep);
        }
        CHECK_NEAR((analytic - numerical).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    }
}

TEST(ImuAttitudeModel, MagnetometerJacobianMatchesCentralDifferences) {
    std::mt19937 generator(37);
    std::normal_distribution<double> normal(0.0, 1.0);
    constexpr double kStep = 1e-6;
    for (int trial = 0; trial < 10; ++trial) {
        const Eigen::VectorXd state = MakeState(Eigen::Vector3d(
            normal(generator), normal(generator), normal(generator)));
        const Eigen::Matrix<double, 3, 4> analytic =
            kalman::imu::MagnetometerJacobian(state);
        Eigen::Matrix<double, 3, 4> numerical =
            Eigen::Matrix<double, 3, 4>::Zero();
        for (int column = 0; column < 4; ++column) {
            Eigen::VectorXd forward = state;
            Eigen::VectorXd backward = state;
            forward(column) += kStep;
            backward(column) -= kStep;
            numerical.col(column) =
                (kalman::imu::MagnetometerPrediction(forward) -
                 kalman::imu::MagnetometerPrediction(backward)) /
                (2.0 * kStep);
        }
        CHECK_NEAR((analytic - numerical).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    }
}

TEST(ImuAttitudeModel, AccelerometerReadsGravityWithTheRightMagnitude) {
    const Eigen::Vector3d euler(20.0 / kDegrees, -35.0 / kDegrees,
                                70.0 / kDegrees);
    const Eigen::VectorXd state = MakeState(euler);
    const Eigen::Vector3d prediction =
        kalman::imu::AccelerometerPrediction(state);
    const Eigen::Vector3d expected =
        kalman::QuaternionToRotationMatrix(state.segment<4>(kQuaternionOffset))
            .transpose() *
        Eigen::Vector3d(0.0, 0.0, -kalman::imu::kGravity);
    CHECK_NEAR((prediction - expected).norm(), 0.0, 1e-12);
    CHECK_NEAR(prediction.norm(), kalman::imu::kGravity, 1e-12);
}

// ---------------------------------------------------------------------------
// Observability: measured, not asserted in prose.
// ---------------------------------------------------------------------------
TEST(ImuAttitudeObservability, GravityConstrainsTwoDirectionsInstantaneously) {
    // The accelerometer produces three numbers that only ever contain two
    // independent pieces of attitude information: the direction of gravity. So
    // *instantaneously* the measurement Jacobian has rank 2 in attitude space
    // (plus the radial quaternion direction, which is not an attitude), and one
    // attitude direction is invisible. That direction is yaw.
    const Eigen::Vector4d attitude =
        kalman::imu::QuaternionFromEulerZyx(Eigen::Vector3d(0.2, 0.15, 0.5));
    Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
    state.segment<4>(kQuaternionOffset) = attitude;
    Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> svd(
        kalman::imu::AccelerometerJacobian(state));
    const Eigen::Vector3d singular_values = svd.singularValues();  // 3 values
    CHECK_TRUE(singular_values(0) > 1.0);
    CHECK_TRUE(singular_values(1) > 1.0);
    CHECK_TRUE(singular_values(2) < 1e-9 * singular_values(0));  // rank 2

    // The invisible attitude direction is a rotation about the world vertical:
    // it changes the heading and leaves the measurement exactly unchanged.
    const double angle = 5.0 / kDegrees;
    const Eigen::Vector4d world_yaw_increment(std::cos(0.5 * angle), 0.0, 0.0,
                                              std::sin(0.5 * angle));
    const Eigen::Vector4d yawed = kalman::QuaternionNormalized(
        kalman::QuaternionMultiply(world_yaw_increment, attitude));
    const Eigen::Vector3d before =
        kalman::QuaternionToRotationMatrix(attitude).transpose() *
        Eigen::Vector3d(0.0, 0.0, -kalman::imu::kGravity);
    const Eigen::Vector3d after =
        kalman::QuaternionToRotationMatrix(yawed).transpose() *
        Eigen::Vector3d(0.0, 0.0, -kalman::imu::kGravity);
    CHECK_TRUE((after - before).norm() < 1e-12);
    const double yaw_angle = std::abs(
        kalman::EulerDifference(kalman::QuaternionToEulerZyx(yawed),
                                kalman::QuaternionToEulerZyx(attitude))(2));
    CHECK_NEAR(yaw_angle, angle, 1e-9);
}

TEST(ImuAttitudeObservability, MagnetometerAddsAThirdIndependentDirection) {
    // Gravity alone fixes two attitude directions, so the accelerometer
    // Jacobian has rank 2. Adding a magnetometer - a second world-frame
    // direction that is not parallel to gravity - makes three directions
    // measurable, so the stacked Jacobian reaches rank 3 and heading stops
    // being invisible. This is checked instantaneously, at one attitude, where
    // the statement is unambiguous.
    const Eigen::Vector4d attitude =
        kalman::imu::QuaternionFromEulerZyx(Eigen::Vector3d(0.2, 0.15, 0.5));
    Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
    state.segment<4>(kQuaternionOffset) = attitude;

    Eigen::Matrix<double, 6, 4> stacked;
    stacked.topRows<3>() = kalman::imu::AccelerometerJacobian(state);
    stacked.bottomRows<3>() = kalman::imu::MagnetometerJacobian(state);
    const Eigen::JacobiSVD<Eigen::Matrix<double, 6, 4>> svd(stacked);
    const Eigen::Matrix<double, 4, 1> singular_values = svd.singularValues();
    // Rank 3 (three attitude directions) plus the radial direction: the third
    // singular value is large, the fourth is zero.
    CHECK_TRUE(singular_values(2) > 1.0);
    CHECK_TRUE(singular_values(3) < 1e-9 * singular_values(0));

    // Without the magnetometer the third direction is missing.
    const Eigen::JacobiSVD<Eigen::Matrix<double, 3, 4>> gravity_only(
        kalman::imu::AccelerometerJacobian(state));
    CHECK_TRUE(gravity_only.singularValues()(2) <
               1e-9 * gravity_only.singularValues()(0));
}

// ---------------------------------------------------------------------------
// End to end, including static calibration.
// ---------------------------------------------------------------------------
namespace {

struct TrackingResult {
    Eigen::Vector3d rms_error_deg = Eigen::Vector3d::Zero();
    Eigen::Vector3d final_error_deg = Eigen::Vector3d::Zero();
    double yaw_drift_deg = 0.0;
    double max_covariance_asymmetry = 0.0;
    double min_covariance_eigenvalue = 0.0;
    int refused_steps = 0;
    Eigen::Vector3d gyro_bias_residual = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias_residual = Eigen::Vector3d::Zero();
    Eigen::Vector4d attitude_at_final_window =
        Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
};

TrackingResult RunTracking(bool use_magnetometer,
                           double magnetometer_start_time, double duration) {
    kalman::imu::SensorParameters sensors;
    const double dt = sensors.sample_time;
    const int steps = static_cast<int>(duration / dt);
    const int calibration_samples = static_cast<int>(3.0 / dt);
    const kalman::imu::Calibration calibration =
        kalman::imu::CalibrateFromStatic(&sensors, calibration_samples, 4242u);
    kalman::imu::SimulatedImu imu(sensors, 987654u);

    const Eigen::Vector3d truth_euler(-5.0 / kDegrees, 10.0 / kDegrees,
                                      30.0 / kDegrees);
    Eigen::Vector4d truth_quaternion =
        kalman::imu::QuaternionFromEulerZyx(truth_euler);

    Eigen::Matrix4d initial_covariance =
        std::pow(2.0 / kDegrees, 2) * Eigen::Matrix4d::Identity();
    kalman::ExtendedKalmanFilter filter(
        StateFromQuaternionForTest(calibration.attitude), initial_covariance);
    const Eigen::Matrix4d process_noise =
        kalman::imu::ProcessNoiseCovariance(sensors, dt);
    const Eigen::Matrix3d accelerometer_noise =
        kalman::imu::AccelerometerNoiseCovariance(sensors);
    const Eigen::Matrix3d magnetometer_noise =
        kalman::imu::MagnetometerNoiseCovariance(sensors);

    TrackingResult result;
    result.gyro_bias_residual = calibration.gyro_bias - sensors.gyro_bias;
    result.accel_bias_residual = calibration.accel_bias - sensors.accel_bias;
    Eigen::Vector3d error_square_sum = Eigen::Vector3d::Zero();
    int counted = 0;
    double yaw_error_at_final_start = 0.0;
    const double final_start = duration - 5.0;

    for (int step = 0; step < steps; ++step) {
        const double time = step * dt;
        const Eigen::Vector3d true_rate = kalman::imu::TrueAngularRate(time);
        truth_quaternion = kalman::QuaternionNormalized(
            kalman::imu::PropagateExact(truth_quaternion, true_rate, dt));
        const Eigen::Vector3d gyro =
            imu.SampleGyro(true_rate) - calibration.gyro_bias;
        const Eigen::Vector3d accelerometer = imu.SampleAccelerometer(
            truth_quaternion, kalman::imu::TrueLinearAcceleration(time));

        if (!filter.Predict(
                [&gyro, dt](const Eigen::VectorXd& x) {
                    return kalman::imu::Transition(x, gyro, dt);
                },
                process_noise,
                [](const Eigen::VectorXd& x) {
                    return kalman::imu::TransitionJacobian(x);
                })) {
            ++result.refused_steps;
            continue;
        }
        if (!filter.Update(kalman::imu::AccelerometerPrediction, accelerometer,
                           accelerometer_noise,
                           kalman::imu::AccelerometerJacobian)) {
            ++result.refused_steps;
            continue;
        }
        if (use_magnetometer && time >= magnetometer_start_time) {
            const Eigen::Vector3d magnetometer =
                imu.SampleMagnetometer(truth_quaternion);
            if (!filter.Update(
                    [](const Eigen::VectorXd& x) {
                        return kalman::imu::MagnetometerPrediction(x);
                    },
                    magnetometer, magnetometer_noise,
                    [](const Eigen::VectorXd& x) {
                        return kalman::imu::MagnetometerJacobian(x);
                    })) {
                ++result.refused_steps;
                continue;
            }
        }

        Eigen::VectorXd normalized = filter.state();
        normalized.segment<4>(kQuaternionOffset) = kalman::QuaternionNormalized(
            normalized.segment<4>(kQuaternionOffset));
        filter.set_state(normalized);

        result.max_covariance_asymmetry =
            std::max(result.max_covariance_asymmetry,
                     (filter.covariance() - filter.covariance().transpose())
                         .cwiseAbs()
                         .maxCoeff());
        const Eigen::SelfAdjointEigenSolver<Eigen::Matrix4d> solver(
            (filter.covariance() + filter.covariance().transpose()) * 0.5);
        result.min_covariance_eigenvalue =
            step == 0 ? solver.eigenvalues()(0)
                      : std::min(result.min_covariance_eigenvalue,
                                 static_cast<double>(solver.eigenvalues()(0)));

        const Eigen::Vector3d error =
            kalman::EulerDifference(
                kalman::imu::EstimateEuler(filter.state()),
                kalman::imu::EstimateEuler(
                    StateFromQuaternionForTest(truth_quaternion))) *
            kDegrees;
        if (time >= final_start) {
            error_square_sum += error.cwiseProduct(error);
            ++counted;
        }
        if (std::abs(time - final_start) < 0.5 * dt) {
            yaw_error_at_final_start = error(2);
            result.attitude_at_final_window = truth_quaternion;
        }
    }

    result.rms_error_deg =
        (error_square_sum / static_cast<double>(counted)).cwiseSqrt();
    result.final_error_deg =
        kalman::EulerDifference(
            kalman::imu::EstimateEuler(filter.state()),
            kalman::imu::EstimateEuler(
                StateFromQuaternionForTest(truth_quaternion))) *
        kDegrees;
    result.yaw_drift_deg = result.final_error_deg(2) - yaw_error_at_final_start;
    return result;
}

}  // namespace

TEST(ImuAttitudeEstimation, StaticCalibrationRecoversTheGyroBias) {
    kalman::imu::SensorParameters sensors;
    const int samples = 300;  // 3 s
    const kalman::imu::Calibration calibration =
        kalman::imu::CalibrateFromStatic(&sensors, samples, 4242u);
    CHECK_TRUE(calibration.valid);
    // Averaging 300 samples of 0.01 rad/s noise leaves ~0.0006 rad/s, so the
    // calibrated bias must be within a few millirad/s of the truth.
    CHECK_TRUE((calibration.gyro_bias - sensors.gyro_bias).norm() < 0.003);
    // And the calibrated attitude must be within a fraction of a degree of
    // level (the device is at rest, so gravity defines roll and pitch).
    const Eigen::Vector3d euler =
        kalman::QuaternionToEulerZyx(calibration.attitude);
    CHECK_TRUE(std::abs(euler(0)) < 1.5 / kDegrees);
    CHECK_TRUE(std::abs(euler(1)) < 1.5 / kDegrees);
}

TEST(ImuAttitudeEstimation, TracksRollAndPitchThroughAManoeuvre) {
    const TrackingResult result = RunTracking(false, 0.0, 20.0);
    CHECK_EQ(result.refused_steps, 0);

    // Roll and pitch are observable through gravity, so the tilt error has a
    // floor set by the sensor noise and by the linear acceleration the model
    // does not contain (which is declared in R, see the attitude model):
    //
    //   tilt error ~ (accel noise + vibration + unmodelled acceleration) / g
    //
    // The bound below is four times that floor, i.e. it is loose on purpose. A
    // tight bound is not a better test here: the same deterministic run differs
    // by ~0.5 degrees between AppleClang and GCC 13 (different floating-point
    // contraction and vectorization), which is enough to flip a 2 degree
    // threshold. What the test should pin down is that the filter tracks tilt
    // *at all* and tracks it far better than yaw, which the ratio checks do.
    const kalman::imu::SensorParameters sensors;
    const double floor_deg =
        (sensors.accel_noise_sigma + sensors.vibration_sigma +
         sensors.accel_bias_residual_sigma) /
        kalman::imu::kGravity * kDegrees;
    CHECK_TRUE(result.rms_error_deg(0) < 4.0 * floor_deg);
    CHECK_TRUE(result.rms_error_deg(1) < 4.0 * floor_deg);
    // Tilt is an order of magnitude better than heading (which has no
    // reference at all): a stable, compiler-independent comparison.
    CHECK_TRUE(result.rms_error_deg(0) < result.rms_error_deg(2));
    CHECK_TRUE(result.rms_error_deg(1) < result.rms_error_deg(2));

    // The covariance stays a valid covariance for all 2000 steps.
    CHECK_TRUE(result.max_covariance_asymmetry < 1e-12);
    CHECK_TRUE(result.min_covariance_eigenvalue > 0.0);
}

TEST(ImuAttitudeEstimation, YawDriftsWithoutAHeadingReferenceAndStopsWithOne) {
    // Roll and pitch are held by gravity; heading is not held by anything. With
    // a gyro + accelerometer pair the residual calibration error therefore
    // shows up as a pure heading drift.
    const TrackingResult imu_only = RunTracking(false, 0.0, 20.0);
    CHECK_TRUE(imu_only.rms_error_deg(0) < 4.0);
    CHECK_TRUE(imu_only.rms_error_deg(1) < 4.0);
    CHECK_TRUE(std::abs(imu_only.yaw_drift_deg) > 0.3);

    // The robust statement is a comparison, not a hand-computed number: heading
    // accumulates error over the whole run because nothing corrects it, while
    // roll and pitch are pulled back to vertical by gravity at every sample.
    // The heading error therefore ends up several times larger than the tilt
    // errors, even though the gyro noise affecting all three axes is identical.
    CHECK_TRUE(std::abs(imu_only.final_error_deg(2)) >
               3.0 * std::max(std::abs(imu_only.final_error_deg(0)),
                              std::abs(imu_only.final_error_deg(1))));

    // Turning the magnetometer on gives heading a reference. (Run from the
    // start rather than mid-run: switching a sensor on mid-flight produces a
    // transient, which is a separate effect from drift.)
    const TrackingResult fused = RunTracking(true, 0.0, 20.0);
    CHECK_TRUE(std::abs(fused.yaw_drift_deg) < 0.2);
    CHECK_TRUE(std::abs(fused.yaw_drift_deg) <
               std::abs(imu_only.yaw_drift_deg) / 3.0);
    CHECK_TRUE(fused.rms_error_deg(0) < 2.0);
    CHECK_TRUE(fused.rms_error_deg(1) < 2.0);
}
