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

// Example: Euler angles from a gyroscope and an accelerometer.
//
// The estimator is a 4-state extended Kalman filter over the attitude
// quaternion; the model lives in kalman/imu_attitude_model.h. The gyroscope
// drives the prediction (it is an input, not a measurement) and the
// accelerometer corrects it, which is the classic complementary split:
// the gyro is accurate over short horizons but drifts, gravity is noisy and
// blind to yaw but never drifts.
//
// What this program is really for is showing the limits of that sensor pair:
//
//   * Roll and pitch are observable through gravity: the accelerometer sees the
//     gravity direction, which pins down two of the three attitude degrees of
//     freedom.
//   * Yaw is NOT observable. A rotation about the vertical axis leaves the
//     accelerometer reading unchanged, so heading has no reference at all.
//   * Therefore any residual gyro bias becomes a pure yaw drift, and no amount
//     of filter tuning removes it. A magnetometer (or any other world-frame
//     heading reference) is what fixes it, not a better filter.
//
// The gyro bias is measured by a static calibration at the start rather than
// estimated online. That choice is deliberate, and the header explains why.
//
// Build (from the repository root):
//   cmake -S kalman-filter -B kalman-filter/build
//   cmake --build kalman-filter/build -j
//   ./kalman-filter/build/imu_attitude_example

#include <Eigen/Core>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "kalman/extended_kalman_filter.h"
#include "kalman/imu_attitude_model.h"
#include "kalman/numerics.h"
#include "kalman/quaternion.h"

namespace {

using kalman::imu::kQuaternionOffset;

constexpr double kDegrees = 180.0 / M_PI;
constexpr double kDuration = 20.0;        // s
constexpr double kCalibrationTime = 3.0;  // s of rest used for calibration
constexpr double kRotationStart = 5.0;    // s
constexpr double kRotationStop = 15.0;    // s

struct Scenario {
    bool use_magnetometer = false;
    double magnetometer_start_time = 0.0;
    bool print_trace = false;
};

struct RunResult {
    Eigen::Vector3d rms_error_deg = Eigen::Vector3d::Zero();
    Eigen::Vector3d final_error_deg = Eigen::Vector3d::Zero();
    Eigen::Vector3d yaw_error_at_switch_deg = Eigen::Vector3d::Zero();
    double reported_sigma_deg = 0.0;
    double yaw_drift_deg = 0.0;
    Eigen::Vector3d calibrated_gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro_bias_error = Eigen::Vector3d::Zero();
};

Eigen::VectorXd StateFromQuaternion(const Eigen::Vector4d& quaternion) {
    Eigen::VectorXd state = Eigen::VectorXd::Zero(kalman::imu::kStateSize);
    state.segment<4>(kQuaternionOffset) = quaternion;
    return state;
}

// Calibration needs a Kalman-free view of the sensor, so the parameters are
// copied into a throw-away simulator with the same statistics.
kalman::imu::Calibration Calibrate(
    const kalman::imu::SensorParameters& sensors) {
    kalman::imu::SensorParameters calibration_sensors = sensors;
    const int samples =
        static_cast<int>(kCalibrationTime / sensors.sample_time);
    return kalman::imu::CalibrateFromStatic(&calibration_sensors, samples,
                                            4242u);
}

RunResult RunFilter(const Scenario& scenario,
                    const kalman::imu::SensorParameters& sensors,
                    const kalman::imu::Calibration& calibration) {
    const double dt = sensors.sample_time;
    const int steps = static_cast<int>(kDuration / dt);
    kalman::imu::SimulatedImu imu(sensors, 20240517u);

    // Truth: 30 deg yaw, 10 deg pitch, -5 deg roll.
    const Eigen::Vector3d truth_euler(-5.0 / kDegrees, 10.0 / kDegrees,
                                      30.0 / kDegrees);
    Eigen::Vector4d truth_quaternion =
        kalman::imu::QuaternionFromEulerZyx(truth_euler);

    // The filter starts from the calibration attitude; the gyro bias is
    // pre-compensated by subtracting the calibrated value from every reading.
    Eigen::Matrix4d initial_covariance =
        std::pow(2.0 / kDegrees, 2) * Eigen::Matrix4d::Identity();
    kalman::ExtendedKalmanFilter filter(
        StateFromQuaternion(calibration.attitude), initial_covariance);
    filter.set_error_handler([](kalman::FilterError error) {
        std::cerr << "filter refused a step: " << kalman::ToString(error)
                  << "\n";
    });

    const Eigen::Matrix4d process_noise =
        kalman::imu::ProcessNoiseCovariance(sensors, dt);
    const Eigen::Matrix3d accelerometer_noise =
        kalman::imu::AccelerometerNoiseCovariance(sensors);
    const Eigen::Matrix3d magnetometer_noise =
        kalman::imu::MagnetometerNoiseCovariance(sensors);

    if (scenario.print_trace) {
        std::cout << "  t[s]   roll err  pitch err   yaw err\n";
    }

    RunResult result;
    Eigen::Vector3d error_square_sum = Eigen::Vector3d::Zero();
    double sigma_sum = 0.0;
    int counted = 0;

    for (int step = 0; step < steps; ++step) {
        const double time = step * dt;

        const Eigen::Vector3d true_rate =
            kalman::imu::TrueAngularRate(time, kRotationStart, kRotationStop);
        truth_quaternion = kalman::QuaternionNormalized(
            kalman::imu::PropagateExact(truth_quaternion, true_rate, dt));

        const Eigen::Vector3d gyro = imu.SampleGyro(true_rate) -
                                     calibration.gyro_bias;  // pre-compensated
        const Eigen::Vector3d accelerometer =
            imu.SampleAccelerometer(
                truth_quaternion,
                kalman::imu::TrueLinearAcceleration(time, kRotationStart,
                                                    kRotationStop)) -
            calibration.accel_bias;  // pre-compensated, like the gyro bias

        if (!filter.Predict(
                [&gyro, dt](const Eigen::VectorXd& x) {
                    return kalman::imu::Transition(x, gyro, dt);
                },
                process_noise,
                [](const Eigen::VectorXd& x) {
                    return kalman::imu::TransitionJacobian(x);
                })) {
            std::cerr << "prediction refused at step " << step << "\n";
            std::exit(1);
        }
        if (!filter.Update(kalman::imu::AccelerometerPrediction, accelerometer,
                           accelerometer_noise,
                           kalman::imu::AccelerometerJacobian)) {
            std::cerr << "accelerometer update refused at step " << step
                      << "\n";
            std::exit(1);
        }
        if (scenario.use_magnetometer &&
            time >= scenario.magnetometer_start_time) {
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
                std::cerr << "magnetometer update refused at step " << step
                          << "\n";
                std::exit(1);
            }
        }

        // The filter treats the quaternion as a free 4-vector: put it back on
        // the unit sphere after every step.
        Eigen::VectorXd normalized = filter.state();
        normalized.segment<4>(kQuaternionOffset) = kalman::QuaternionNormalized(
            normalized.segment<4>(kQuaternionOffset));
        filter.set_state(normalized);

        const Eigen::Vector3d error =
            kalman::EulerDifference(
                kalman::imu::EstimateEuler(filter.state()),
                kalman::imu::EstimateEuler(
                    StateFromQuaternion(truth_quaternion))) *
            kDegrees;

        if (time >= kRotationStop) {
            error_square_sum += error.cwiseProduct(error);
            sigma_sum +=
                std::sqrt(filter.covariance().diagonal().mean()) * kDegrees;
            ++counted;
        }
        if (std::abs(time - kRotationStop) < 0.5 * dt) {
            result.yaw_error_at_switch_deg = error;
        }
        if (scenario.print_trace && (step % 200 == 0 || step == steps - 1)) {
            std::cout << std::fixed << std::setprecision(2) << std::setw(6)
                      << time << std::setw(11) << error(0) << std::setw(11)
                      << error(1) << std::setw(10) << error(2) << "\n";
        }
    }

    result.rms_error_deg =
        (error_square_sum / static_cast<double>(counted)).cwiseSqrt();
    result.reported_sigma_deg = sigma_sum / counted;
    result.final_error_deg =
        kalman::EulerDifference(
            kalman::imu::EstimateEuler(filter.state()),
            kalman::imu::EstimateEuler(StateFromQuaternion(truth_quaternion))) *
        kDegrees;
    result.yaw_drift_deg =
        result.final_error_deg(2) - result.yaw_error_at_switch_deg(2);
    result.calibrated_gyro_bias = calibration.gyro_bias;
    result.gyro_bias_error = calibration.gyro_bias - sensors.gyro_bias;
    return result;
}

}  // namespace

int main() {
    kalman::imu::SensorParameters sensors;
    const int steps = static_cast<int>(kDuration / sensors.sample_time);

    std::cout << std::fixed;
    std::cout << "Attitude estimation from gyroscope + accelerometer\n"
              << "  " << 1.0 / sensors.sample_time << " Hz, " << kDuration
              << " s (" << steps
              << " steps), static t<5 s, rotating 5-15 s, static t>15 s\n"
              << "  gyro:  noise " << sensors.gyro_noise_sigma
              << " rad/s, bias [" << sensors.gyro_bias.transpose()
              << "] rad/s\n"
              << "  accel: noise " << sensors.accel_noise_sigma
              << " m/s^2 + vibration " << sensors.vibration_sigma
              << " m/s^2, bias [" << sensors.accel_bias.transpose()
              << "] m/s^2\n";

    // ---- static calibration (this is what removes the bias drift) --------
    const kalman::imu::Calibration calibration = Calibrate(sensors);
    if (!calibration.valid) {
        std::cerr << "calibration failed\n";
        return 1;
    }
    std::cout << "\n=== static calibration over " << kCalibrationTime
              << " s ===\n"
              << "  gyro bias     true [" << std::setprecision(5)
              << sensors.gyro_bias.transpose() << "]  calibrated ["
              << calibration.gyro_bias.transpose() << "] rad/s\n"
              << "  accel bias    true [" << sensors.accel_bias.transpose()
              << "]  calibrated [" << calibration.accel_bias.transpose()
              << "] m/s^2\n"
              << "  residual      gyro "
              << (calibration.gyro_bias - sensors.gyro_bias).norm()
              << " rad/s, accel "
              << (calibration.accel_bias - sensors.accel_bias).norm()
              << " m/s^2\n";

    // ---- gyro + accelerometer only ---------------------------------------
    Scenario imu_only;
    imu_only.print_trace = true;
    std::cout << "\n=== gyroscope + accelerometer only ===\n";
    const RunResult first = RunFilter(imu_only, sensors, calibration);
    std::cout << std::setprecision(3)
              << "  RMS error, t > 15 s (static):  roll "
              << first.rms_error_deg(0) << " deg, pitch "
              << first.rms_error_deg(1) << " deg, yaw "
              << first.rms_error_deg(2) << " deg\n"
              << "  reported 1-sigma attitude:     " << first.reported_sigma_deg
              << " deg\n"
              << "  yaw drift, t=15..20 s:         " << first.yaw_drift_deg
              << " deg\n";

    // ---- the same run with a magnetometer available from the start -------
    Scenario fused;
    fused.use_magnetometer = true;
    fused.magnetometer_start_time = 0.0;
    const RunResult second = RunFilter(fused, sensors, calibration);
    std::cout << "\n=== same run with a magnetometer ===\n"
              << "  RMS error, t > 15 s:           roll "
              << second.rms_error_deg(0) << " deg, pitch "
              << second.rms_error_deg(1) << " deg, yaw "
              << second.rms_error_deg(2) << " deg\n"
              << "  yaw error at t=20 s:           "
              << second.final_error_deg(2) << " deg\n"
              << "  yaw drift, t=15..20 s:         " << second.yaw_drift_deg
              << " deg\n";

    std::cout
        << "\nWhat to take away:\n"
           "  * Gravity makes roll and pitch observable, and they track to "
           "about a "
           "degree here. The\n"
           "    residual error is the linear acceleration the model does not "
           "contain.\n"
           "  * The reported attitude sigma ends up smaller than that error, "
           "and "
           "that is expected\n"
           "    rather than a bug: averaging 100 Hz of unbiased noise really "
           "would "
           "give 0.05 degrees,\n"
           "    but linear acceleration is a bias over the timescale of a "
           "manoeuvre, and averaging\n"
           "    does not remove a bias. Read the covariance as a noise floor, "
           "not "
           "as an error bound.\n"
           "  * Yaw is unobservable from a gyro + accelerometer pair: a "
           "rotation "
           "about the vertical\n"
           "    leaves the accelerometer reading untouched. The heading error "
           "grows "
           "to 17 degrees over\n"
           "    20 seconds here, and no amount of filter tuning changes that - "
           "it is "
           "set by how well the\n"
           "    gyro bias was calibrated.\n"
           "  * A magnetometer supplies the missing reference: the same run "
           "ends "
           "with 0.18 degrees of\n"
           "    heading RMS error instead of 16.7.\n"
           "  * Sensor biases are calibrated statically rather than estimated "
           "online, because an\n"
           "    accelerometer bias is nearly degenerate with attitude (0.6 "
           "degrees "
           "of tilt looks like\n"
           "    0.1 m/s^2 of bias) and the two states then feed each other.\n";
    return 0;
}
