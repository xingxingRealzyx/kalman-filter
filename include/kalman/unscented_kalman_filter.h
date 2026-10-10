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

#ifndef KALMAN_UNSCENTED_KALMAN_FILTER_H_
#define KALMAN_UNSCENTED_KALMAN_FILTER_H_

// Unscented Kalman filter (UKF) with the scaled unscented transform.
//
//   x[k] = f(x[k-1]) + w[k-1],   w ~ N(0, Q)
//   y[k] = h(x[k])   + v[k],     v ~ N(0, R)
//
// Instead of linearizing f and h, the UKF propagates 2n+1 deterministically
// chosen sigma points through the true nonlinear functions and recovers mean
// and covariance from the transformed points. No Jacobians are needed, and the
// estimate is accurate to at least second order (third order for symmetric
// distributions), which is why it tends to beat the EKF on strongly curved
// models such as bearings-only tracking or attitude kinematics.
//
// Sigma point scaling follows the standard formulation with
//   lambda = alpha^2 (n + kappa) - n,   kappa = 3 - n,
//   W0^m = lambda / (n + lambda),
//   W0^c = lambda / (n + lambda) + (1 - alpha^2 + beta),
//   Wi^m = Wi^c = 1 / (2 (n + lambda)),  i = 1..2n
// with alpha = 1e-3, beta = 2, kappa = 3 - n by default. Note that the tiny
// alpha keeps the sigma points close to the mean, which is what makes the
// transform numerically well behaved for large n.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <functional>

#include "kalman/numerics.h"

namespace kalman {

class UnscentedKalmanFilter {
 public:
    using StateVector = Eigen::VectorXd;
    using StateMatrix = Eigen::MatrixXd;
    using MeasurementVector = Eigen::VectorXd;
    using MeasurementMatrix = Eigen::MatrixXd;

    using StateTransitionFunction =
        std::function<StateVector(const StateVector&)>;
    using MeasurementFunction =
        std::function<MeasurementVector(const StateVector&)>;

    UnscentedKalmanFilter();
    UnscentedKalmanFilter(const StateVector& initial_state,
                          const StateMatrix& initial_covariance);

    // ---- configuration ---------------------------------------------------
    double alpha() const { return alpha_; }
    // alpha is usually left at 1e-3: smaller spreads the sigma points further
    // from the mean, which is more accurate for very nonlinear models but can
    // amplify noise. Set it before the first Predict().
    void set_alpha(double alpha);
    void set_beta(double beta) { beta_ = beta; }
    // kappa defaults to 3 - n; set it explicitly only if you know why.
    void set_kappa(double kappa) {
        kappa_ = kappa;
        kappa_is_default_ = false;
    }

    // ---- accessors -------------------------------------------------------
    const StateVector& state() const { return x_; }
    const StateMatrix& covariance() const { return P_; }
    int state_dimension() const { return static_cast<int>(x_.size()); }

    const MeasurementVector& innovation() const { return innovation_; }
    const MeasurementMatrix& innovation_covariance() const {
        return innovation_covariance_;
    }
    const Eigen::MatrixXd& kalman_gain() const { return kalman_gain_; }

    bool Reset(const StateVector& initial_state,
               const StateMatrix& initial_covariance);
    void set_state(const StateVector& state) { x_ = state; }
    void set_covariance(const StateMatrix& covariance);

    // ---- prediction ------------------------------------------------------
    // Returns false (and leaves the filter untouched) when the sigma point
    // decomposition fails or a propagated point is not finite.
    bool Predict(const StateTransitionFunction& transition,
                 const StateMatrix& process_noise);

    // Convenience overloads for a linear model. They are named differently from
    // the nonlinear ones on purpose: an overload set taking both a matrix and a
    // std::function would make every `Predict(lambda, q)` call ambiguous.
    // Mostly useful for tests and for cross-checking the UKF against the linear
    // Kalman filter.
    bool PredictLinear(const StateMatrix& state_transition,
                       const StateMatrix& process_noise);

    // ---- measurement update ---------------------------------------------
    bool Correct(const MeasurementFunction& measurement,
                 const MeasurementVector& y,
                 const StateMatrix& measurement_noise);

    bool CorrectLinear(const MeasurementMatrix& measurement_matrix,
                       const MeasurementVector& y,
                       const StateMatrix& measurement_noise);

 private:
    // Deterministic sigma points and unscented weights for the current n.
    bool GenerateSigmaPoints(const StateVector& mean,
                             const StateMatrix& covariance,
                             Eigen::MatrixXd* sigma_points);

    void UpdateWeights();

    StateVector x_;
    StateMatrix P_;
    MeasurementVector innovation_;
    MeasurementMatrix innovation_covariance_;
    Eigen::MatrixXd kalman_gain_;
    Eigen::VectorXd weights_mean_;
    Eigen::VectorXd weights_covariance_;

    double alpha_ = 1e-3;
    double beta_ = 2.0;
    double kappa_ = 0.0;
    bool kappa_is_default_ = true;
};

}  // namespace kalman

#endif  // KALMAN_UNSCENTED_KALMAN_FILTER_H_
