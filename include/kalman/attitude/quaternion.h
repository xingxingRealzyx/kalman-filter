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

#ifndef KALMAN_QUATERNION_H_
#define KALMAN_QUATERNION_H_

// Minimal quaternion and rotation helpers for attitude estimation.
//
// Conventions used throughout this library:
//   * The quaternion is [w, x, y, z], scalar part first, and represents the
//     rotation from the body frame to the world frame: v_world = R(q) v_body.
//   * The angular rate w is expressed in the body frame.
//   * Euler angles are ZYX (yaw about world z, then pitch about the new y, then
//     roll about the new x), returned as [roll, pitch, yaw]. This is the
//     "aerospace" sequence, and it has a singularity at pitch = +/- 90 degrees.
//
// Quaternions are used for the *state* precisely because Euler angles are
// singular there; Euler angles are only produced as output for humans.

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>

namespace kalman {

// Hamilton product, same order as Eigen's AngleAxis composition:
// (p * q) applies q first, then p.
inline Eigen::Vector4d QuaternionMultiply(const Eigen::Vector4d& p,
                                          const Eigen::Vector4d& q) {
    return Eigen::Vector4d(
        p(0) * q(0) - p(1) * q(1) - p(2) * q(2) - p(3) * q(3),
        p(0) * q(1) + p(1) * q(0) + p(2) * q(3) - p(3) * q(2),
        p(0) * q(2) - p(1) * q(3) + p(2) * q(0) + p(3) * q(1),
        p(0) * q(3) + p(1) * q(2) - p(2) * q(1) + p(3) * q(0));
}

// Matrix of left multiplication by q, i.e. QuaternionLeftMatrix(q) * p ==
// QuaternionMultiply(q, p). This is what makes the attitude propagation a
// linear operation during one sample: dq/dt = 0.5 * Omega(w) q.
inline Eigen::Matrix4d QuaternionLeftMatrix(const Eigen::Vector4d& q) {
    Eigen::Matrix4d matrix;
    matrix << q(0), -q(1), -q(2), -q(3),  //
        q(1), q(0), -q(3), q(2),          //
        q(2), q(3), q(0), -q(1),          //
        q(3), -q(2), q(1), q(0);
    return matrix;
}

// Matrix of right multiplication by q, i.e. QuaternionRightMatrix(p) * q ==
// QuaternionMultiply(p, q). This is the one that belongs in attitude
// propagation, because Eigen composes rotations as (p * q) = "q first, then p",
// so q_{k+1} = q_k * increment means "rotate about the *body* axes" - which is
// exactly what a gyroscope measures. Beware: the widely copied
// dq/dt = 0.5 Omega(w) q with Omega = [[0, -w^T], [w, -[w]x]] is the
// left-multiplication form and corresponds to the conjugate rotation, i.e. to
// world-frame rates. Mixing the two conventions halves the effective rate.
inline Eigen::Matrix4d QuaternionRightMatrix(const Eigen::Vector4d& q) {
    Eigen::Matrix4d matrix;
    matrix << q(0), -q(1), -q(2), -q(3),  //
        q(1), q(0), q(3), -q(2),          //
        q(2), -q(3), q(0), q(1),          //
        q(3), q(2), -q(1), q(0);
    return matrix;
}

inline Eigen::Vector4d QuaternionConjugate(const Eigen::Vector4d& q) {
    return Eigen::Vector4d(q(0), -q(1), -q(2), -q(3));
}

// Projects a quaternion onto the unit sphere. A Kalman filter treats the
// quaternion as a free 4-vector, so it drifts off the sphere; renormalizing
// after every step is the standard fix (it makes the covariance slightly
// optimistic along the radial direction, which is harmless because that
// direction carries no information about attitude).
inline Eigen::Vector4d QuaternionNormalized(const Eigen::Vector4d& q,
                                            double epsilon = 1e-12) {
    const double norm = q.norm();
    if (!(norm > epsilon)) {
        return Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);
    }
    return q / norm;
}

// Rotation matrix of a quaternion: v_world = R(q) v_body.
//
// The quaternion is normalized first, on purpose. Eigen's
// Quaterniond::toRotationMatrix() does NOT normalize: feeding it a quaternion
// of norm s scales the "rotation" by s^2, which is not a rotation matrix at all
// (|R| - I grows by 3.5 for a norm of 2.2). A Kalman filter keeps the
// quaternion only approximately on the unit sphere, so the difference matters:
// it silently breaks the agreement between this model function and its analytic
// Jacobian, and the filter then drifts. Normalizing here makes the model a true
// function of attitude, and makes the Jacobian below the exact derivative of
// what this function computes.
inline Eigen::Matrix3d QuaternionToRotationMatrix(const Eigen::Vector4d& q) {
    const Eigen::Quaterniond quaternion(q(0), q(1), q(2), q(3));
    return quaternion.normalized().toRotationMatrix();
}

// Jacobian of R(q) * v with respect to the four quaternion components, where
// R(q) is the rotation of the normalized quaternion.
//
// This is a central difference of QuaternionToRotationMatrix, on purpose, and
// the header earns an explanation rather than a formula. A closed form for this
// 3x4 Jacobian was written by hand four times while building this library, and
// every version was wrong in a different way (a sign on the scalar column, two
// missing terms, then a radial projection applied on the wrong side). Each
// wrong version still passed the "does it look like a rotation Jacobian" smell
// test, and each one silently corrupted the covariance of the attitude filter.
//
// Two checks pin a correct implementation down, and this one passes both to
// machine precision:
//
//   * J * q = 0. R does not depend on the quaternion norm, so the derivative
//     must be orthogonal to the radial direction.
//   * J equals the central difference of the model function along the
//     renormalized path, which is the path a filter actually takes.
//
// Central differences cost a handful of rotation matrix evaluations per call -
// irrelevant next to the rest of a filter step, and the code is short enough to
// be obviously correct. tests/quaternion_test.cpp asserts both properties.
inline Eigen::Matrix<double, 3, 4> RotationMatrixJacobian(
    const Eigen::Vector4d& q, const Eigen::Vector3d& v) {
    constexpr double kStep = 1e-7;  // balanced for double precision here
    Eigen::Matrix<double, 3, 4> jacobian;
    Eigen::Vector4d perturbed = q;
    for (int column = 0; column < 4; ++column) {
        perturbed(column) = q(column) + kStep;
        const Eigen::Vector3d forward =
            QuaternionToRotationMatrix(perturbed) * v;
        perturbed(column) = q(column) - kStep;
        const Eigen::Vector3d backward =
            QuaternionToRotationMatrix(perturbed) * v;
        perturbed(column) = q(column);
        jacobian.col(column) = (forward - backward) / (2.0 * kStep);
    }
    return jacobian;
}

// Predicts the measurement of a world-frame vector w (gravity, magnetic field)
// as seen by a body-mounted sensor: a = R(q)^T w.
inline Eigen::Vector3d BodyFrameVector(const Eigen::Vector4d& q,
                                       const Eigen::Vector3d& world_vector) {
    return QuaternionToRotationMatrix(q).transpose() * world_vector;
}

// Jacobian of BodyFrameVector with respect to the quaternion. R(q)^T = R(q^-1)
// and the conjugate is linear, so this is the rotation Jacobian of the
// conjugated quaternion with the vector part negated. That identity is checked
// against central differences in tests/quaternion_test.cpp.
inline Eigen::Matrix<double, 3, 4> BodyFrameVectorJacobian(
    const Eigen::Vector4d& q, const Eigen::Vector3d& world_vector) {
    Eigen::Matrix<double, 3, 4> jacobian =
        RotationMatrixJacobian(QuaternionConjugate(q), world_vector);
    jacobian.rightCols<3>() *= -1.0;
    return jacobian;
}

// ZYX Euler angles [roll, pitch, yaw] from a body-to-world quaternion.
// Returns the standard expressions, which are singular at |pitch| = 90 deg.
inline Eigen::Vector3d QuaternionToEulerZyx(const Eigen::Vector4d& q) {
    const double w = q(0);
    const double x = q(1);
    const double y = q(2);
    const double z = q(3);

    Eigen::Vector3d euler;
    euler(0) = std::atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y));
    const double sin_pitch = 2.0 * (w * y - z * x);
    euler(1) = std::asin(std::max(-1.0, std::min(1.0, sin_pitch)));
    euler(2) = std::atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z));
    return euler;
}

// Difference between two Euler angle triples, wrapped component-wise into
// (-pi, pi]. Use this for error statistics, never a plain subtraction.
inline Eigen::Vector3d EulerDifference(const Eigen::Vector3d& first,
                                       const Eigen::Vector3d& second) {
    constexpr double kTwoPi = 2.0 * 3.14159265358979323846;
    Eigen::Vector3d difference;
    for (int i = 0; i < 3; ++i) {
        difference(i) = std::remainder(first(i) - second(i), kTwoPi);
    }
    return difference;
}

}  // namespace kalman

#endif  // KALMAN_QUATERNION_H_
