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

#include "kalman/attitude/quaternion.h"

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <random>

#include "tests/testing.h"

namespace {

Eigen::Vector4d RandomUnitQuaternion(std::mt19937* generator) {
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::Vector4d q;
    q << normal(*generator), normal(*generator), normal(*generator),
        normal(*generator);
    return kalman::QuaternionNormalized(q);
}

}  // namespace

TEST(Quaternion, MultiplyMatchesEigenProduct) {
    std::mt19937 generator(20240517);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d p = RandomUnitQuaternion(&generator);
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        const Eigen::Vector4d product = kalman::QuaternionMultiply(p, q);

        // Same convention as Eigen: (p * q) applies q first, then p.
        const Eigen::Quaterniond p_eigen(p(0), p(1), p(2), p(3));
        const Eigen::Quaterniond q_eigen(q(0), q(1), q(2), q(3));
        const Eigen::Quaterniond expected = p_eigen * q_eigen;
        CHECK_NEAR((product - Eigen::Vector4d(expected.w(), expected.x(),
                                              expected.y(), expected.z()))
                       .norm(),
                   0.0, 1e-14);
    }
}

TEST(Quaternion, LeftMatrixReproducesMultiplication) {
    std::mt19937 generator(11);
    for (int trial = 0; trial < 10; ++trial) {
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        const Eigen::Vector4d p = RandomUnitQuaternion(&generator);
        CHECK_NEAR((kalman::QuaternionLeftMatrix(q) * p -
                    kalman::QuaternionMultiply(q, p))
                       .norm(),
                   0.0, 1e-14);
    }
}

TEST(Quaternion, RotationMatrixMatchesEigen) {
    std::mt19937 generator(23);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        const Eigen::Quaterniond reference(q(0), q(1), q(2), q(3));
        CHECK_NEAR((kalman::QuaternionToRotationMatrix(q) -
                    reference.toRotationMatrix())
                       .norm(),
                   0.0, 1e-14);
        // Orthonormal with determinant +1.
        const Eigen::Matrix3d rotation = kalman::QuaternionToRotationMatrix(q);
        CHECK_NEAR(
            (rotation * rotation.transpose() - Eigen::Matrix3d::Identity())
                .norm(),
            0.0, 1e-14);
        CHECK_NEAR(rotation.determinant(), 1.0, 1e-14);
    }
}

TEST(Quaternion, RotationJacobianMatchesCentralDifferences) {
    // This is the formula that is easy to get wrong (the scalar column carries
    // a sign the vector columns do not), so it is pinned against finite
    // differences rather than trusted.
    std::mt19937 generator(31);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        Eigen::Vector3d v;
        v << normal(generator), normal(generator), normal(generator);

        Eigen::Matrix<double, 3, 4> numerical;
        constexpr double kStep = 1e-6;
        for (int column = 0; column < 4; ++column) {
            Eigen::Vector4d forward = q;
            Eigen::Vector4d backward = q;
            forward(column) += kStep;
            backward(column) -= kStep;
            numerical.col(column) =
                (kalman::QuaternionToRotationMatrix(forward) * v -
                 kalman::QuaternionToRotationMatrix(backward) * v) /
                (2.0 * kStep);
        }
        const Eigen::Matrix<double, 3, 4> analytic =
            kalman::RotationMatrixJacobian(q, v);
        CHECK_NEAR((analytic - numerical).cwiseAbs().maxCoeff(), 0.0, 1e-6);
    }
}

TEST(Quaternion, RotationJacobianAnnihilatesTheRadialDirection) {
    // The rotation does not depend on the quaternion norm, so a correct
    // Jacobian must map the radial direction q to zero. This is the cheapest
    // and most decisive check on the formula: it fails loudly for a wrong sign,
    // while a finite-difference comparison against a normalized propagation can
    // hide it.
    std::mt19937 generator(71);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        Eigen::Vector3d v;
        v << normal(generator), normal(generator), normal(generator);
        CHECK_NEAR((kalman::RotationMatrixJacobian(q, v) * q).norm(), 0.0,
                   1e-7);
        CHECK_NEAR((kalman::BodyFrameVectorJacobian(q, v) * q).norm(), 0.0,
                   1e-7);
    }
}

TEST(Quaternion, BodyFrameJacobianMatchesCentralDifferences) {
    std::mt19937 generator(41);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int trial = 0; trial < 20; ++trial) {
        const Eigen::Vector4d q = RandomUnitQuaternion(&generator);
        Eigen::Vector3d world;
        world << normal(generator), normal(generator), normal(generator);

        // The measurement model must agree with the rotation matrix form.
        CHECK_NEAR((kalman::BodyFrameVector(q, world) -
                    kalman::QuaternionToRotationMatrix(q).transpose() * world)
                       .norm(),
                   0.0, 1e-14);

        Eigen::Matrix<double, 3, 4> numerical;
        constexpr double kStep = 1e-6;
        for (int column = 0; column < 4; ++column) {
            Eigen::Vector4d forward = q;
            Eigen::Vector4d backward = q;
            forward(column) += kStep;
            backward(column) -= kStep;
            numerical.col(column) = (kalman::BodyFrameVector(forward, world) -
                                     kalman::BodyFrameVector(backward, world)) /
                                    (2.0 * kStep);
        }
        CHECK_NEAR((kalman::BodyFrameVectorJacobian(q, world) - numerical)
                       .cwiseAbs()
                       .maxCoeff(),
                   0.0, 1e-6);
    }
}

TEST(Quaternion, EulerConversionRoundTrips) {
    // Build a rotation from known Euler angles, convert it to a quaternion, and
    // check that the conversion back reproduces the angles.
    const double degrees = M_PI / 180.0;
    const double roll_values[] = {-170.0, -90.0, -30.0, 0.0, 45.0, 179.0};
    const double pitch_values[] = {-80.0, -20.0, 0.0, 15.0, 70.0};
    const double yaw_values[] = {-175.0, -45.0, 0.0, 60.0, 178.0};

    for (const double roll : roll_values) {
        for (const double pitch : pitch_values) {
            for (const double yaw : yaw_values) {
                Eigen::Vector3d euler(roll * degrees, pitch * degrees,
                                      yaw * degrees);
                const Eigen::Matrix3d rotation =
                    (Eigen::AngleAxisd(euler(2), Eigen::Vector3d::UnitZ()) *
                     Eigen::AngleAxisd(euler(1), Eigen::Vector3d::UnitY()) *
                     Eigen::AngleAxisd(euler(0), Eigen::Vector3d::UnitX()))
                        .toRotationMatrix();
                const Eigen::Quaterniond quaternion(rotation);
                Eigen::Vector4d q(quaternion.w(), quaternion.x(),
                                  quaternion.y(), quaternion.z());

                const Eigen::Vector3d recovered =
                    kalman::QuaternionToEulerZyx(q);
                CHECK_NEAR(kalman::EulerDifference(recovered, euler)
                               .cwiseAbs()
                               .maxCoeff(),
                           0.0, 1e-9);
            }
        }
    }
}

TEST(Quaternion, EulerDifferenceWrapsAcrossPi) {
    // -179 deg and +179 deg are two degrees apart, not 358.
    const double degrees = M_PI / 180.0;
    Eigen::Vector3d first(-179.0 * degrees, 0.0, 0.0);
    Eigen::Vector3d second(179.0 * degrees, 0.0, 0.0);
    CHECK_NEAR(kalman::EulerDifference(first, second)(0), 2.0 * degrees, 1e-12);
    CHECK_NEAR(kalman::EulerDifference(second, first)(0), -2.0 * degrees,
               1e-12);
}

TEST(Quaternion, NormalizationHandlesDegenerateInput) {
    Eigen::Vector4d almost_zero = Eigen::Vector4d::Zero();
    const Eigen::Vector4d result = kalman::QuaternionNormalized(almost_zero);
    CHECK_NEAR(result(0), 1.0, 0.0);
    CHECK_NEAR(result.tail<3>().norm(), 0.0, 0.0);

    std::mt19937 generator(53);
    for (int trial = 0; trial < 10; ++trial) {
        Eigen::Vector4d scaled = 100.0 * RandomUnitQuaternion(&generator);
        CHECK_NEAR(kalman::QuaternionNormalized(scaled).norm(), 1.0, 1e-14);
    }
}
