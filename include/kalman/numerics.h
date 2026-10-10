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

#ifndef KALMAN_NUMERICS_H_
#define KALMAN_NUMERICS_H_

// Small numerical helpers shared by all filters in this library.
//
// Everything here is defensive on purpose: a Kalman filter that silently
// propagates a non-symmetric or non-positive-semidefinite covariance does not
// fail loudly, it just starts to trust itself more and more until the estimate
// walks away from the truth. These helpers make that failure mode detectable
// (and, where cheap, recoverable) at every step.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>

namespace kalman {

// Returns true when every coefficient is finite (no NaN, no +/-Inf).
template <typename Derived>
bool AllFinite(const Eigen::MatrixBase<Derived>& matrix) {
    return matrix.allFinite();
}

// Forces exact symmetry by averaging A and A^T. Covariance propagation in
// floating point loses symmetry at the last-bit level; left alone, that error
// accumulates and can make a Cholesky factorization fail.
//
// The transpose is evaluated into a temporary first. Writing
// `A = (A + A.transpose()) * 0.5` looks equivalent but is not: Eigen evaluates
// the assignment coefficient by coefficient, so the transpose would read
// already-overwritten elements of A and quietly corrupt the result.
template <typename Derived>
void SymmetrizeInPlace(Eigen::MatrixBase<Derived>& matrix) {
    const typename Derived::PlainObject transpose =
        matrix.derived().transpose();
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index column = 0; column < matrix.cols(); ++column) {
            matrix.derived()(row, column) =
                (matrix.derived()(row, column) + transpose(row, column)) * 0.5;
        }
    }
}

// Returns true when the symmetric part of `matrix` has no significantly
// negative eigenvalue. `tolerance` is relative to the largest eigenvalue.
template <typename Derived>
bool IsPositiveSemidefinite(const Eigen::MatrixBase<Derived>& matrix,
                            double tolerance = 1e-9) {
    if (!AllFinite(matrix)) {
        return false;
    }
    Eigen::SelfAdjointEigenSolver<typename Derived::PlainObject> solver(
        ((matrix.derived() + matrix.derived().transpose()) * 0.5).eval(),
        Eigen::EigenvaluesOnly);
    if (solver.info() != Eigen::Success) {
        return false;
    }
    const double largest = solver.eigenvalues().cwiseAbs().maxCoeff();
    const double scale = largest > 0.0 ? largest : 1.0;
    return solver.eigenvalues().minCoeff() >= -tolerance * scale;
}

// Adds `jitter` times the identity to `matrix` until it is positive definite
// (up to `max_attempts` tries, each growing by 10x). Returns true when the
// matrix is positive definite on exit. Used as a last resort before inverting a
// nearly singular innovation covariance. Note that any jitter is a *bias*: it
// is always preferable to solve the exact system, which is what
// SolveKalmanGain below does.
template <typename Derived>
bool RegularizeInPlace(Eigen::MatrixBase<Derived>& matrix,
                       double jitter = 1e-12, int max_attempts = 8) {
    using Scalar = typename Derived::Scalar;
    if (!AllFinite(matrix)) {
        return false;
    }
    SymmetrizeInPlace(matrix);
    const Scalar scale =
        std::max(static_cast<Scalar>(matrix.derived().cwiseAbs().maxCoeff()),
                 static_cast<Scalar>(1));
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        Eigen::LLT<typename Derived::PlainObject> llt(matrix.derived());
        if (llt.info() == Eigen::Success) {
            return true;
        }
        matrix.derived().diagonal().array() +=
            static_cast<Scalar>(jitter * scale);
    }
    return false;
}

// Computes K = X S^-1 (written as the transpose of the solve for S K^T = X^T),
// where S is the innovation covariance. Returns an empty matrix when no
// reliable gain exists, and the caller must then refuse the step.
//
// The plain Cholesky solve is preferred because its result is exact: adding
// jitter to S biases every covariance the filter reports. A singular S can fail
// in two ways that need different answers:
//
//   * S is singular and X has a component along the null space. The gain does
//     not exist (0/0) and "no result" is the only honest answer.
//   * S is singular but X lies entirely inside its range. The gain is well
//     defined and zero in that direction, and returning zero is correct.
//
// Cholesky reports failure for both, so the decision is made by checking the
// residual of the computed gain instead: a gain satisfying S K^T = X^T is kept,
// while one that does not (Eigen can return infinities from a factorization
// that reported success) triggers a jittered retry. If that retry is what
// produces a finite gain, the step is still reported as unusable — the
// alternative is a silently biased covariance.
template <typename DerivedS, typename DerivedX>
Eigen::Matrix<typename DerivedS::Scalar, Eigen::Dynamic, Eigen::Dynamic>
SolveKalmanGain(const Eigen::MatrixBase<DerivedS>& s,
                const Eigen::MatrixBase<DerivedX>& x) {
    using Scalar = typename DerivedS::Scalar;
    using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

    Matrix s_eval(s);
    const Matrix x_eval(x);
    if (s_eval.rows() != s_eval.cols() || s_eval.rows() != x_eval.cols() ||
        !AllFinite(s_eval) || !AllFinite(x_eval)) {
        return Matrix();
    }
    SymmetrizeInPlace(s_eval);

    const Scalar scale =
        std::max(static_cast<Scalar>(s_eval.cwiseAbs().maxCoeff()),
                 static_cast<Scalar>(1));
    const Scalar tolerance = static_cast<Scalar>(1e-12) * scale;

    const auto residual_is_small = [&](const Matrix& gain) {
        if (!AllFinite(gain)) {
            return false;
        }
        const Vector residual = (s_eval * gain.transpose() - x_eval.transpose())
                                    .cwiseAbs()
                                    .rowwise()
                                    .maxCoeff();
        return residual.maxCoeff() <= tolerance;
    };

    Matrix gain = s_eval.llt().solve(x_eval.transpose()).transpose();
    if (residual_is_small(gain)) {
        return gain;
    }

    Matrix perturbed = s_eval;
    for (int attempt = 0; attempt < 12; ++attempt) {
        perturbed = s_eval;
        perturbed.diagonal().array() +=
            static_cast<Scalar>(scale * std::pow(10.0, -15.0 + attempt));
        gain = perturbed.llt().solve(x_eval.transpose()).transpose();
        if (residual_is_small(gain)) {
            return Matrix();  // Salvaged only by regularization: refuse the
                              // step.
        }
    }
    return Matrix();
}

// Lifts every angle-like component of a vector into (-pi, pi].
//
// Two overloads so that both plain vectors and expressions such as
// `state.head<1>()` work. For an expression the coefficients are written
// through to the underlying matrix, which is what a caller expects from a
// function with "InPlace" in its name.
//
// std::remainder maps a half turn to -pi (it rounds the quotient to even), so
// the result is nudged to +pi: every angle then has exactly one representation,
// and a filter that reports -pi does not look like it jumped a full turn.
template <typename Derived>
void NormalizeAnglesInPlace(Eigen::MatrixBase<Derived>& vector) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        auto& value = vector.derived()(i);
        value = std::remainder(value, kTwoPi);
        if (value == -kPi) {
            value = kPi;
        }
    }
}

template <typename Derived>
void NormalizeAnglesInPlace(Eigen::MatrixBase<Derived>&& vector) {
    constexpr double kPi = 3.14159265358979323846;
    constexpr double kTwoPi = 2.0 * kPi;
    for (Eigen::Index i = 0; i < vector.size(); ++i) {
        auto& value = vector.derived()(i);
        value = std::remainder(value, kTwoPi);
        if (value == -kPi) {
            value = kPi;
        }
    }
}

// Shortest signed distance between two angles, in (-pi, pi]. Use this instead
// of `a - b` whenever you compare two angles: the plain difference reports a
// 359 degree error where the true error is 1 degree.
double AngularDifference(double first, double second);

}  // namespace kalman

#endif  // KALMAN_NUMERICS_H_
