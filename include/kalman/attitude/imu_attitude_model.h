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

#ifndef KALMAN_IMU_ATTITUDE_MODEL_H_
#define KALMAN_IMU_ATTITUDE_MODEL_H_

// Attitude estimation model for a gyroscope + accelerometer (+ optional
// magnetometer) IMU, ready to hand to kalman::ExtendedKalmanFilter.
//
//   state x = [q_w q_x q_y q_z | b_gx b_gy b_gz | b_ax b_ay b_az]
//             |body->world quaternion|  gyro bias     accel bias
//
//   gyro   w_m = w + b_g + n_g                      -> prediction input
//   accel  a_m = R(q)^T (g - a_linear) + b_a + n_a  -> measurement
//   mag    m_m = R(q)^T m_world + n_m               -> measurement (optional)
//
// The gyro is an *input*, not a measurement: it enters the model callbacks,
// which is why the transition and Jacobian functions here take the gyro reading
// as an argument. Everything in this header is deterministic - sensor sampling
// lives in the sensor struct at the bottom, so simulations are reproducible.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>

#include "kalman/attitude/quaternion.h"
#include "kalman/numerics.h"

namespace kalman {
namespace imu {

// ------------------------------------------------------------- state layout
// The estimated state is the attitude alone:
//
//   x = [q_w q_x q_y q_z]      body-to-world unit quaternion
//
// Gyro and accelerometer biases are deliberately NOT states here. Two reasons,
// both learned the hard way while building this example:
//
//   * An accelerometer bias is very nearly degenerate with attitude: a tilt of
//     0.6 degrees changes the measured gravity by the same 0.1 m/s^2 that a
//     typical bias does. Putting both in the state lets them feed each other,
//     and a filter that is merely slightly overconfident then walks away.
//   * Sensor biases are constant for a run, so they are better measured once,
//     during a static calibration, than tracked online. That is what real
//     implementations do, and it is what `CalibrationFromStatic` below is for.
//
// The consequence is honest and worth measuring: an uncalibrated gyro bias is a
// pure yaw (heading) drift, which no accelerometer can see.
constexpr int kQuaternionOffset = 0;  // 4 components
constexpr int kStateSize = 4;

// ------------------------------------------------------------------- physics
constexpr double kGravity = 9.81;  // m/s^2

// A unit world-frame magnetic field in the x-z plane. Only the direction
// matters: the field magnitude carries no attitude information, so the
// magnetometer is treated as normalized.
inline Eigen::Vector3d WorldMagneticField(double declination = 0.0) {
    return Eigen::Vector3d(std::cos(declination), std::sin(declination), 0.0);
}

// ------------------------------------------------------------ sensor model
struct SensorParameters {
    double gyro_noise_sigma = 0.01;       // rad/s, white
    double gyro_bias_walk_sigma = 1e-4;   // rad/s per step
    double accel_noise_sigma = 0.05;      // m/s^2, white
    double accel_bias_walk_sigma = 1e-3;  // m/s^2 per step
    double vibration_sigma = 0.30;        // m/s^2, correlated, unmodelled
    // Everything the accelerometer measures that is not gravity or white
    // noise: the bias left by calibration, and - much larger - the linear
    // acceleration of the vehicle the IMU is bolted to. It is declared here
    // rather than estimated because the filter cannot separate it from tilt,
    // and because averaging 100 Hz of *biased* samples does not reduce a bias:
    // without this term the reported attitude covariance comes out at 0.05
    // degrees while the actual error is a degree, i.e. the filter lies.
    double accel_bias_residual_sigma = 0.30;  // m/s^2
    double magnetometer_noise_sigma = 0.01;   // normalized field units
    double magnetometer_bias_walk_sigma =
        2e-3;  // normalized field units per step
    Eigen::Vector3d gyro_bias = Eigen::Vector3d(0.020, -0.015, 0.010);
    Eigen::Vector3d accel_bias = Eigen::Vector3d(0.10, -0.05, 0.08);
    Eigen::Vector3d magnetometer_bias = Eigen::Vector3d::Zero();
    double magnetic_declination = 0.0;
    double sample_time = 0.01;  // 100 Hz
};

// --------------------------------------------------------------- prediction
// Quaternion increment over one sample for a body-frame rate: the rotation by
// angle |w| Ts about the axis w/|w|, which is exact for a constant rate over
// the sample. Eigen composes rotations as (p * q) = "q first, then p", so the
// increment multiplies from the right for a body-frame rate.
inline Eigen::Vector4d QuaternionIncrement(const Eigen::Vector3d& angular_rate,
                                           double sample_time) {
    const double angle = angular_rate.norm() * sample_time;
    if (angle < 1e-12) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    const Eigen::Vector3d axis = angular_rate / angular_rate.norm();
    const Eigen::Quaterniond increment(Eigen::AngleAxisd(angle, axis));
    return Eigen::Vector4d(increment.w(), increment.x(), increment.y(),
                           increment.z());
}

// One sample of the state transition: the quaternion advances with the
// bias-corrected rate, the biases are random walks and so stay put in the mean.
inline Eigen::VectorXd Transition(const Eigen::VectorXd& x,
                                  const Eigen::Vector3d& bias_corrected_rate,
                                  double sample_time) {
    Eigen::VectorXd next = x;
    next.segment<4>(kQuaternionOffset) = QuaternionNormalized(
        QuaternionRightMatrix(x.segment<4>(kQuaternionOffset)) *
        QuaternionIncrement(bias_corrected_rate, sample_time));
    return next;
}

// Jacobian of Transition with respect to the quaternion.
//
//   d q_{k+1}/d q = N * L(q)   with L the right-multiplication matrix (the
//                              body-rate convention) and N the projector onto
//                              the unit sphere, I - q q^T / |q|^2, because the
//                              propagation renormalizes. The step size does not
//                              enter: it is inside the increment.
//
// Renormalization is why the projector belongs here: a finite difference of the
// normalized map measures the radial component too, and quietly disagrees with
// the tangent-space answer. Verified against the unnormalized map times the
// projector in tests/quaternion_test.cpp.
inline Eigen::Matrix4d TransitionJacobian(const Eigen::VectorXd& x) {
    const Eigen::Vector4d q = x.segment<4>(kQuaternionOffset);
    const double squared_norm = q.squaredNorm();
    Eigen::Matrix4d projector = Eigen::Matrix4d::Identity();
    if (squared_norm > 1e-12) {
        projector -= q * q.transpose() / squared_norm;
    }
    return projector * QuaternionRightMatrix(q);
}

// Continuous-time noise on the gyro, discretized over one sample: the white
// noise enters the quaternion as (Ts/2)^2 sigma_g^2 per axis.
inline Eigen::Matrix4d ProcessNoiseCovariance(const SensorParameters& sensors,
                                              double sample_time) {
    const double variance = 0.25 * sample_time * sample_time *
                            sensors.gyro_noise_sigma * sensors.gyro_noise_sigma;
    return variance * Eigen::Matrix4d::Identity();
}

// -------------------------------------------------------------- measurement
// The accelerometer measures specific force: gravity minus linear
// acceleration, expressed in the body frame. Linear acceleration is *not* in
// the state, which is exactly why it is the dominant error source for this
// filter; the vibration term in R is how the user tells the filter about it.
inline Eigen::VectorXd AccelerometerPrediction(const Eigen::VectorXd& x) {
    return BodyFrameVector(x.segment<4>(kQuaternionOffset),
                           Eigen::Vector3d(0.0, 0.0, -kGravity));
}

inline Eigen::Matrix<double, 3, 4> AccelerometerJacobian(
    const Eigen::VectorXd& x) {
    return BodyFrameVectorJacobian(x.segment<4>(kQuaternionOffset),
                                   Eigen::Vector3d(0.0, 0.0, -kGravity));
}

inline Eigen::VectorXd MagnetometerPrediction(const Eigen::VectorXd& x,
                                              double declination = 0.0) {
    const Eigen::Vector4d q = x.segment<4>(kQuaternionOffset);
    return BodyFrameVector(q, WorldMagneticField(declination));
}

inline Eigen::Matrix<double, 3, 4> MagnetometerJacobian(
    const Eigen::VectorXd& x, double declination = 0.0) {
    return BodyFrameVectorJacobian(x.segment<4>(kQuaternionOffset),
                                   WorldMagneticField(declination));
}

// Measurement noise for the accelerometer: white sensor noise, the vibration
// the model does not explain, and the residual bias left by calibration.
// Reported to the filter as one number because the filter cannot tell them
// apart - and because R is the only place a linear acceleration can be declared
// when it is not a state.
inline Eigen::Matrix3d AccelerometerNoiseCovariance(
    const SensorParameters& sensors) {
    const double variance =
        sensors.accel_noise_sigma * sensors.accel_noise_sigma +
        sensors.vibration_sigma * sensors.vibration_sigma +
        sensors.accel_bias_residual_sigma * sensors.accel_bias_residual_sigma;
    return variance * Eigen::Matrix3d::Identity();
}

inline Eigen::Matrix3d MagnetometerNoiseCovariance(
    const SensorParameters& sensors) {
    return (sensors.magnetometer_noise_sigma *
            sensors.magnetometer_noise_sigma) *
           Eigen::Matrix3d::Identity();
}

// ------------------------------------------------------------------ helpers
// Attitude as seen by the user: ZYX Euler angles in radians.
inline Eigen::Vector3d EstimateEuler(const Eigen::VectorXd& state) {
    return QuaternionToEulerZyx(state.segment<4>(kQuaternionOffset));
}

inline Eigen::Vector4d QuaternionFromEulerZyx(const Eigen::Vector3d& euler) {
    const Eigen::Quaterniond quaternion(
        Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ()) *
        Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX()));
    return Eigen::Vector4d(quaternion.w(), quaternion.x(), quaternion.y(),
                           quaternion.z());
}

// Exact propagation of a true attitude over one sample (used by the simulator).
inline Eigen::Vector4d PropagateExact(const Eigen::Vector4d& q,
                                      const Eigen::Vector3d& angular_rate,
                                      double sample_time) {
    const double rate_norm = angular_rate.norm();
    Eigen::Vector3d axis = Eigen::Vector3d::UnitX();
    if (rate_norm > 1e-12) {
        axis =
            angular_rate / rate_norm;  // named local: Eigen expressions of two
                                       // different types cannot share a ternary
    }
    const Eigen::Quaterniond rotation(
        Eigen::AngleAxisd(rate_norm * sample_time, axis));
    const Eigen::Quaterniond current(q(0), q(1), q(2), q(3));
    // Eigen composes as (a * b) = "b first, then a", so a rotation about the
    // *body* axes multiplies from the left. (For reference: gyro+accelerometer
    // runs with the order swapped drift by tens of degrees, because the truth
    // and the model then disagree about which frame the rate is expressed in.)
    const Eigen::Quaterniond next = rotation * current;
    return Eigen::Vector4d(next.w(), next.x(), next.y(), next.z());
}

// --------------------------------------------------------------- simulation
// Deterministic sensor emulation: one instance owns its random engine, so a
// fixed seed reproduces a run exactly.
class SimulatedImu {
 public:
    explicit SimulatedImu(const SensorParameters& parameters, unsigned int seed)
        : parameters_(parameters), generator_(seed) {}

    Eigen::Vector3d SampleGyro(const Eigen::Vector3d& true_rate) {
        Eigen::Vector3d reading = true_rate + parameters_.gyro_bias;
        for (int axis = 0; axis < 3; ++axis) {
            reading(axis) += parameters_.gyro_noise_sigma * Normal();
        }
        return reading;
    }

    // `linear_acceleration` is the true acceleration in the world frame, which
    // the accelerometer cannot distinguish from gravity.
    Eigen::Vector3d SampleAccelerometer(
        const Eigen::Vector4d& true_quaternion,
        const Eigen::Vector3d& linear_acceleration) {
        const Eigen::Vector3d specific_force =
            Eigen::Vector3d(0.0, 0.0, -kGravity) - linear_acceleration;
        Eigen::Vector3d reading =
            QuaternionToRotationMatrix(true_quaternion).transpose() *
                specific_force +
            parameters_.accel_bias;
        for (int axis = 0; axis < 3; ++axis) {
            reading(axis) += parameters_.accel_noise_sigma * Normal() +
                             parameters_.vibration_sigma * Normal();
        }
        return reading;
    }

    Eigen::Vector3d SampleMagnetometer(const Eigen::Vector4d& true_quaternion) {
        Eigen::Vector3d reading =
            QuaternionToRotationMatrix(true_quaternion).transpose() *
                WorldMagneticField(parameters_.magnetic_declination) +
            parameters_.magnetometer_bias;
        for (int axis = 0; axis < 3; ++axis) {
            reading(axis) += parameters_.magnetometer_noise_sigma * Normal();
        }
        return reading;
    }

    const SensorParameters& parameters() const { return parameters_; }

 private:
    double Normal() {
        return std::normal_distribution<double>(0.0, 1.0)(generator_);
    }

    SensorParameters parameters_;
    std::mt19937 generator_;
};

// Static calibration: with the device at rest, the gyro reading IS the bias and
// the accelerometer reading IS the gravity direction. Both are averaged over
// `samples` readings. This is the step that makes the filter below work, and
// the reason its state does not need bias terms.
struct Calibration {
    Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_bias = Eigen::Vector3d::Zero();
    Eigen::Vector4d attitude = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    bool valid = false;
};

inline Calibration CalibrateFromStatic(SensorParameters* sensors, int samples,
                                       unsigned int seed) {
    Calibration calibration;
    if (samples <= 0) {
        return calibration;
    }
    // A freshly seeded simulator with the bias already inside it is the honest
    // way to calibrate: the routine only sees what a real one would.
    SimulatedImu imu(*sensors, seed);
    Eigen::Vector3d gyro_sum = Eigen::Vector3d::Zero();
    Eigen::Vector3d accel_sum = Eigen::Vector3d::Zero();
    for (int i = 0; i < samples; ++i) {
        gyro_sum += imu.SampleGyro(Eigen::Vector3d::Zero());
        accel_sum += imu.SampleAccelerometer(
            Eigen::Vector4d(1.0, 0.0, 0.0, 0.0), Eigen::Vector3d::Zero());
    }
    calibration.gyro_bias = gyro_sum / static_cast<double>(samples);
    const Eigen::Vector3d measured_specific_force =
        accel_sum / static_cast<double>(samples);
    // At rest the accelerometer reads gravity plus its own bias, and it cannot
    // tell them apart. Assuming the device is level (or knowing the local
    // gravity direction) is what makes the split possible: the bias is the part
    // of the reading that is not g along the world vertical.
    const Eigen::Vector3d gravity_body = Eigen::Vector3d(
        measured_specific_force(0), measured_specific_force(1),
        measured_specific_force(2) < 0.0 ? -kGravity : kGravity);
    calibration.accel_bias = measured_specific_force - gravity_body;
    // With the bias removed the device is assumed level at calibration time, so
    // the starting attitude is the identity (this is the standard "place it
    // flat and press calibrate" step).
    const Eigen::Vector3d measured = Eigen::Vector3d(0.0, 0.0, -1.0);
    calibration.attitude = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    static_cast<void>(measured);
    calibration.valid = true;
    return calibration;
}

// A test motion that is static first, then excites all three axes, then static
// again. Static segments are what expose yaw drift; the rotating segment is
// what makes the gyro bias observable.
inline Eigen::Vector3d TrueAngularRate(double time, double start = 5.0,
                                       double stop = 15.0) {
    if (time < start || time > stop) {
        return Eigen::Vector3d::Zero();
    }
    const double ramp = std::min(1.0, (time - start) / 1.0) *
                        std::min(1.0, (stop - time) / 1.0);
    Eigen::Vector3d rate;
    rate << 0.60 * std::sin(1.1 * (time - start)),
        0.45 * std::sin(1.7 * (time - start) + 0.7),
        0.30 * std::sin(0.7 * (time - start) + 1.9);
    return ramp * rate;
}

// Linear acceleration profile: zero when static, a modest wobble while moving.
inline Eigen::Vector3d TrueLinearAcceleration(double time, double start = 5.0,
                                              double stop = 15.0) {
    if (time < start || time > stop) {
        return Eigen::Vector3d::Zero();
    }
    // A hand-held device moved at a moderate pace: a few tenths of a g of
    // horizontal acceleration and a gentle vertical component.
    Eigen::Vector3d acceleration;
    acceleration << 0.25 * std::sin(2.0 * time), 0.20 * std::cos(1.3 * time),
        0.15;
    return acceleration;
}

}  // namespace imu
}  // namespace kalman

#endif  // KALMAN_IMU_ATTITUDE_MODEL_H_
