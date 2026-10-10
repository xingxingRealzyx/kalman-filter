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

#ifndef KALMAN_LINEAR_KALMAN_FILTER_H_
#define KALMAN_LINEAR_KALMAN_FILTER_H_

// Linear (ordinary) Kalman filter for the model
//
//   x[k] = A x[k-1] + B u[k-1] + w[k-1],   w ~ N(0, Q)
//   y[k] = C x[k]               + v[k],     v ~ N(0, R)
//
// The filter is header-only and fully fixed-size so the whole recursion can
// live in registers / on the stack on an MCU. The state transition matrix A,
// control matrix B, and measurement matrix C are passed to Predict()/Update()
// rather than being stored, so a time-varying (LTV) model needs no extra API.
//
// Preferred usage (the C++17 form, no temporaries at the call site):
//
//   kalman::LinearKalmanFilter<4, 2> filter(x0, P0);
//   filter.Predict(A, B, u, Q);
//   filter.Correct(C, y, R);

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <cstddef>

#include "kalman/numerics.h"

namespace kalman {

// How the covariance update P = (I - K C) P^- is evaluated.
enum class CovarianceUpdate {
    // Fast, textbook form. Can lose positive definiteness after many steps
    // with extreme conditioning.
    kStandard,
    // Joseph form: P = (I - K C) P^- (I - K C)^T + K R K^T. Costs roughly
    // twice as much but stays positive semidefinite for any gain K, which
    // matters when R is very small or the model is badly scaled. This is the
    // default because "it does not silently break" beats "it is fast".
    kJoseph,
};

namespace internal {

// Symmetrizes and (cheaply) validates a covariance matrix in place. Returns
// false when the matrix is not finite or is not positive semidefinite.
template <typename Derived>
bool SanitizeCovariance(Eigen::MatrixBase<Derived>& matrix) {
    if (!AllFinite(matrix)) {
        return false;
    }
    SymmetrizeInPlace(matrix);
    return IsPositiveSemidefinite(matrix);
}

}  // namespace internal

template <int N, int M, int NU = Eigen::Dynamic, typename Scalar = double>
class LinearKalmanFilter {
 public:
    static_assert(N > 0, "State dimension must be positive");
    static_assert(M > 0, "Measurement dimension must be positive");
    static_assert(NU == Eigen::Dynamic || NU > 0,
                  "Control dimension must be positive or Dynamic");

    using StateVector = Eigen::Matrix<Scalar, N, 1>;
    using MeasurementVector = Eigen::Matrix<Scalar, M, 1>;
    using ControlVector = Eigen::Matrix<Scalar, NU, 1>;
    using StateMatrix = Eigen::Matrix<Scalar, N, N>;
    using MeasurementMatrix = Eigen::Matrix<Scalar, M, N>;
    using ControlMatrix = Eigen::Matrix<Scalar, N, NU>;
    using MeasurementCovariance = Eigen::Matrix<Scalar, M, M>;
    using GainMatrix = Eigen::Matrix<Scalar, N, M>;

    // Default constructor: zero state, zero covariance. Call Reset() before
    // the first Predict()/Update().
    LinearKalmanFilter()
        : x_(StateVector::Zero()),
          P_(StateMatrix::Zero()),
          covariance_update_(CovarianceUpdate::kJoseph) {}

    LinearKalmanFilter(const StateVector& initial_state,
                       const StateMatrix& initial_covariance)
        : x_(initial_state),
          P_(initial_covariance),
          covariance_update_(CovarianceUpdate::kJoseph) {
        internal::SanitizeCovariance(P_);
    }

    // ---- accessors -------------------------------------------------------
    const StateVector& state() const { return x_; }
    const StateMatrix& covariance() const { return P_; }
    CovarianceUpdate covariance_update() const { return covariance_update_; }
    void set_covariance_update(CovarianceUpdate mode) {
        covariance_update_ = mode;
    }

    void set_state(const StateVector& state) { x_ = state; }
    void set_covariance(const StateMatrix& covariance) {
        P_ = covariance;
        internal::SanitizeCovariance(P_);
    }

    // Re-initializes mean and covariance. Returns false if the covariance is
    // not usable (non-finite or non-positive-semidefinite); the filter keeps
    // its previous state in that case.
    bool Reset(const StateVector& initial_state,
               const StateMatrix& initial_covariance) {
        StateMatrix candidate = initial_covariance;
        if (!internal::SanitizeCovariance(candidate)) {
            return false;
        }
        x_ = initial_state;
        P_ = candidate;
        return true;
    }

    // ---- prediction ------------------------------------------------------
    // x^- = A x + B u,  P^- = A P A^T + Q  (Q defaults to zero).
    template <typename DerivedA, typename DerivedB, typename DerivedU>
    void Predict(const Eigen::MatrixBase<DerivedA>& a,
                 const Eigen::MatrixBase<DerivedB>& b,
                 const Eigen::MatrixBase<DerivedU>& u) {
        Predict(a, b, u, StateMatrix::Zero());
    }

    template <typename DerivedA, typename DerivedB, typename DerivedU,
              typename DerivedQ>
    void Predict(const Eigen::MatrixBase<DerivedA>& a,
                 const Eigen::MatrixBase<DerivedB>& b,
                 const Eigen::MatrixBase<DerivedU>& u,
                 const Eigen::MatrixBase<DerivedQ>& q) {
        // Dimensions are checked before any multiplication so that a caller
        // mistake returns quietly instead of tripping an Eigen assertion.
        if (a.rows() != N || a.cols() != N || b.rows() != N ||
            b.cols() != u.size() || u.rows() != b.cols() || u.cols() != 1) {
            return;
        }
        // Explicit construction, not assignment: the inputs are templates, so
        // assignment would trigger Eigen's compile-time size check before the
        // runtime guard above can report the mismatch as a false return.
        const StateMatrix q_eval(q);
        if (q_eval.rows() != N || q_eval.cols() != N || !AllFinite(q_eval) ||
            !IsPositiveSemidefinite(q_eval)) {
            return;  // Refuse to corrupt the filter with an invalid Q.
        }
        x_ = a * x_ + b * u;
        P_ = a * P_ * a.transpose() + q_eval;
        internal::SanitizeCovariance(P_);
    }

    // Autonomy form (no control input).
    template <typename DerivedA>
    void Predict(const Eigen::MatrixBase<DerivedA>& a) {
        Predict(a, StateMatrix::Zero());
    }

    template <typename DerivedA, typename DerivedQ>
    void Predict(const Eigen::MatrixBase<DerivedA>& a,
                 const Eigen::MatrixBase<DerivedQ>& q) {
        if (a.rows() != N || a.cols() != N) {
            return;
        }
        const StateMatrix q_eval(q);
        if (q_eval.rows() != N || q_eval.cols() != N || !AllFinite(q_eval) ||
            !IsPositiveSemidefinite(q_eval)) {
            return;
        }
        x_ = a * x_;
        P_ = a * P_ * a.transpose() + q_eval;
        internal::SanitizeCovariance(P_);
    }

    // ---- measurement update ---------------------------------------------
    // K = P^- C^T (C P^- C^T + R)^-1
    // x = x^- + K (y - C x^-),  P = (I - K C) P^-
    // Returns false when the input is unusable (non-finite measurement,
    // non-finite or non-PSD R, or a singular innovation covariance); the
    // filter then keeps its predicted state.
    template <typename DerivedC, typename DerivedY, typename DerivedR>
    bool Update(const Eigen::MatrixBase<DerivedC>& c,
                const Eigen::MatrixBase<DerivedY>& y,
                const Eigen::MatrixBase<DerivedR>& r) {
        // A mismatch between the measurement vector and the measurement model
        // is a bug in the caller; catch it here rather than deep inside Eigen.
        static_assert(DerivedY::RowsAtCompileTime == Eigen::Dynamic ||
                          DerivedY::RowsAtCompileTime == M,
                      "LinearKalmanFilter<>::Update: the measurement vector y "
                      "must have M rows");
        static_assert(DerivedC::RowsAtCompileTime == Eigen::Dynamic ||
                          DerivedC::RowsAtCompileTime == M,
                      "LinearKalmanFilter<>::Update: the measurement matrix C "
                      "must have M rows");
        if (c.rows() != M || c.cols() != N || y.size() != M || y.cols() != 1 ||
            r.rows() != M || r.cols() != M) {
            return false;
        }
        const MeasurementMatrix c_eval(c);
        const MeasurementVector y_eval(y);
        const MeasurementCovariance r_eval(r);
        if (!AllFinite(c_eval) || !AllFinite(y_eval) || !AllFinite(r_eval) ||
            !IsPositiveSemidefinite(r_eval)) {
            return false;
        }

        // Innovation covariance S = C P^- C^T + R.
        const MeasurementCovariance s =
            c_eval * P_ * c_eval.transpose() + r_eval;

        // K = (P^- C^T) S^-1, solved through S rather than forming S^-1.
        const Eigen::MatrixXd gain =
            SolveKalmanGain(s, P_ * c_eval.transpose());
        if (gain.rows() != N || gain.cols() != M) {
            return false;
        }
        const GainMatrix k = gain;

        const MeasurementVector innovation = y_eval - c_eval * x_;
        x_ = x_ + k * innovation;

        switch (covariance_update_) {
            case CovarianceUpdate::kJoseph: {
                // P = (I - K C) P^- (I - K C)^T + K R K^T. Written as above it
                // is symmetric by construction and PSD for any K.
                const StateMatrix i_kc = StateMatrix::Identity() - k * c_eval;
                P_ = i_kc * P_ * i_kc.transpose() + k * r_eval * k.transpose();
                break;
            }
            case CovarianceUpdate::kStandard:
            default: {
                const StateMatrix i_kc = StateMatrix::Identity() - k * c_eval;
                P_ = i_kc * P_;
                break;
            }
        }
        internal::SanitizeCovariance(P_);
        return true;
    }

    // Convenience form for scalar measurements: UpdateScalar(c_row, y, r).
    template <typename DerivedC>
    bool UpdateScalar(const Eigen::MatrixBase<DerivedC>& c, Scalar y,
                      Scalar r) {
        MeasurementVector y_vector;
        y_vector << y;
        MeasurementCovariance r_matrix;
        r_matrix << r;
        return Update(c, y_vector, r_matrix);
    }

 private:
    StateVector x_;
    StateMatrix P_;
    CovarianceUpdate covariance_update_;
};

}  // namespace kalman

#endif  // KALMAN_LINEAR_KALMAN_FILTER_H_
