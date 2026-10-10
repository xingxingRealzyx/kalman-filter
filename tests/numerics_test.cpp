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

#include "kalman/numerics.h"

#include <Eigen/Core>
#include <cmath>
#include <limits>

#include "tests/testing.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

}  // namespace

TEST(Numerics, SymmetrizeMakesMatrixExactlySymmetric) {
    Eigen::Matrix3d matrix;
    matrix << 1.0, 2.0, 3.0,  //
        4.0, 5.0, 6.0,        //
        7.0, 8.0, 9.0;
    kalman::SymmetrizeInPlace(matrix);
    CHECK_TRUE(matrix == matrix.transpose());  // exact, element by element
    CHECK_NEAR(matrix(0, 1), 3.0, 0.0);
    CHECK_NEAR(matrix(0, 0), 1.0, 0.0);
    CHECK_NEAR(matrix(2, 0), 5.0, 0.0);
}

TEST(Numerics, PositiveSemidefiniteDetection) {
    Eigen::Matrix2d identity = Eigen::Matrix2d::Identity();
    CHECK_TRUE(kalman::IsPositiveSemidefinite(identity));

    // Singular but PSD (a rank-one projection) must be accepted.
    Eigen::Matrix2d projection;
    projection << 1.0, 1.0, 1.0, 1.0;
    CHECK_TRUE(kalman::IsPositiveSemidefinite(projection));

    // Distinctly negative eigenvalue must be rejected.
    Eigen::Matrix2d negative;
    negative << 1.0, 0.0, 0.0, -1.0;
    CHECK_FALSE(kalman::IsPositiveSemidefinite(negative));

    // Non-finite input must be rejected.
    Eigen::Matrix2d not_a_number = Eigen::Matrix2d::Identity();
    not_a_number(1, 1) = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(kalman::IsPositiveSemidefinite(not_a_number));

    // Tiny negative eigenvalue from floating-point round-off is tolerated.
    Eigen::Matrix2d round_off = identity;
    round_off(0, 0) = 1.0;
    round_off(1, 1) = -1e-15;
    CHECK_TRUE(kalman::IsPositiveSemidefinite(round_off));
}

TEST(Numerics, RegularizeMakesSingularMatrixFactorizable) {
    Eigen::Matrix2d singular;
    singular << 1.0, 0.0, 0.0, 0.0;  // rank one: LLT would fail
    CHECK_TRUE(kalman::RegularizeInPlace(singular));
    CHECK_TRUE(kalman::IsPositiveSemidefinite(singular));

    // A matrix with a negative eigenvalue is repaired by the jitter, provided
    // the jitter is large enough relative to the scale.
    Eigen::Matrix2d indefinite;
    indefinite << 1.0, 0.0, 0.0, -1e-6;
    CHECK_TRUE(kalman::RegularizeInPlace(indefinite, 1e-3));

    Eigen::Matrix2d not_a_number = Eigen::Matrix2d::Identity();
    not_a_number(0, 0) = std::numeric_limits<double>::infinity();
    CHECK_FALSE(kalman::RegularizeInPlace(not_a_number));
}

TEST(Numerics, AngleNormalizationWrapsIntoMinusPiPi) {
    Eigen::Vector3d angles;
    angles << 0.5, 3.0 * kPi, -3.5 * kPi;
    kalman::NormalizeAnglesInPlace(angles);
    CHECK_NEAR(angles(0), 0.5, 1e-15);
    // A half turn is always reported as +pi, never -pi.
    CHECK_NEAR(angles(1), kPi, 1e-15);
    // -3.5 pi wraps to +0.5 pi.
    CHECK_NEAR(angles(2), 0.5 * kPi, 1e-15);
    CHECK_TRUE(angles(0) <= kPi && angles(0) > -kPi);
    CHECK_TRUE(angles(1) <= kPi && angles(1) > -kPi);
    CHECK_TRUE(angles(2) <= kPi && angles(2) > -kPi);

    // The boundary is represented exactly once.
    Eigen::Vector2d boundary;
    boundary << kPi, -kPi;
    kalman::NormalizeAnglesInPlace(boundary);
    CHECK_NEAR(boundary(0), kPi, 0.0);
    CHECK_NEAR(boundary(1), kPi, 0.0);

    // Already-normalized input is left alone (no needless round-off).
    Eigen::Vector2d untouched;
    untouched << -1.25, 2.5;
    const Eigen::Vector2d original = untouched;
    kalman::NormalizeAnglesInPlace(untouched);
    CHECK_TRUE((untouched - original).norm() == 0.0);
}

TEST(Numerics, AngularDifferenceTakesTheShortWayRound) {
    CHECK_NEAR(kalman::AngularDifference(0.1, 0.05), 0.05, 1e-15);
    // 350 deg vs 10 deg: the naive difference is 340 deg, the true error is
    // -20.
    const double difference =
        kalman::AngularDifference(-170.0 * kPi / 180.0, 170.0 * kPi / 180.0);
    CHECK_NEAR(difference, 20.0 * kPi / 180.0, 1e-12);
    // Wrapping is antisymmetric.
    CHECK_NEAR(kalman::AngularDifference(3.0, -3.0) +
                   kalman::AngularDifference(-3.0, 3.0),
               0.0, 1e-15);
}

TEST(Numerics, AllFiniteRejectsNanAndInfinity) {
    Eigen::Vector2d good;
    good << 1.0, -2.0;
    CHECK_TRUE(kalman::AllFinite(good));

    Eigen::Vector2d bad = good;
    bad(0) = std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(kalman::AllFinite(bad));
    bad(0) = std::numeric_limits<double>::infinity();
    CHECK_FALSE(kalman::AllFinite(bad));
}
