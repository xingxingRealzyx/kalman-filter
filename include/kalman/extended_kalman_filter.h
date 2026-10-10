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

#ifndef KALMAN_EXTENDED_KALMAN_FILTER_H_
#define KALMAN_EXTENDED_KALMAN_FILTER_H_

// Extended Kalman filter for the nonlinear model
//
//   x[k] = f(x[k-1]) + w[k-1],   w ~ N(0, Q)
//   y[k] = h(x[k])   + v[k],     v ~ N(0, R)
//
// The algorithm is the linear Kalman filter run on the first-order Taylor
// expansion of f and h about the current estimate:
//
//   prediction:  x^- = f(x)
//                F   = df/dx |_x                  (caller-supplied Jacobian)
//                P^- = F P F^T + Q
//   update:      H   = dh/dx |_{x^-}
//                S   = H P^- H^T + R              (innovation covariance)
//                K   = P^- H^T S^-1
//                x   = x^- + K (y - h(x^-))
//                P   = (I - K H) P^- (I - K H)^T + K R K^T     (Joseph)
//
// Sizes are dynamic because user models rarely have compile-time dimensions;
// pass fixed-size lambdas and the compiler still sees through them.
//
// Two details that matter in practice:
//   * The Jacobian is optional. When it is not supplied it is computed by
//     central finite differences of the model function. For hand-written
//     models this is usually the right default: less code, fewer sign errors,
//     and a cost that is negligible next to the model itself.
//   * State components that are angles must be normalized by the model
//     functions themselves (see NormalizeAnglesInPlace in numerics.h);
//     the filter cannot know which components wrap.

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <functional>

#include "kalman/numerics.h"

namespace kalman {

// Why a filter step was refused. A refused step leaves the filter untouched, so
// it is a safe outcome - but a *silently* refused step is not: the estimate
// simply stops improving and a downstream controller keeps consuming a stale
// state. ExtendedKalmanFilter::set_error_handler() exists so that "why" is one
// line of code away.
enum class FilterError {
    kNone = 0,
    kInvalidDimensions,
    kNonFiniteInput,
    kCovarianceNotPositiveSemidefinite,
    kSingularInnovationCovariance,
    kNonFiniteModelOutput,
    // The supplied Jacobian has the wrong shape. This is the one that hurts:
    // StateTransitionJacobian and MeasurementJacobian have the same C++ type
    // (std::function<MatrixXd(const VectorXd&)>), so passing a 1xM measurement
    // Jacobian where a NxN transition Jacobian belongs compiles cleanly.
    kJacobianDimensionMismatch,
    kNonFiniteJacobian,
};

const char* ToString(FilterError error);

class ExtendedKalmanFilter {
 public:
    using StateVector = Eigen::VectorXd;
    using StateMatrix = Eigen::MatrixXd;
    using MeasurementVector = Eigen::VectorXd;
    using MeasurementMatrix = Eigen::MatrixXd;

    // Propagates one sample: x[k] = f(x[k-1]).
    using StateTransitionFunction =
        std::function<StateVector(const StateVector&)>;
    // Jacobian of the transition, df/dx, evaluated at the given state.
    using StateTransitionJacobian =
        std::function<StateMatrix(const StateVector&)>;
    // Predicts the measurement: y_hat = h(x).
    using MeasurementFunction =
        std::function<MeasurementVector(const StateVector&)>;
    // Jacobian of the measurement, dh/dx, evaluated at the given state.
    //
    // Careful: this type is identical to StateTransitionJacobian, so the
    // compiler cannot stop you from passing one where the other is expected.
    // Predict() checks the shape at run time and reports
    // FilterError::kJacobianDimensionMismatch through the error handler.
    using MeasurementJacobian =
        std::function<MeasurementMatrix(const StateVector&)>;

    // Called with the reason whenever a step is refused. Empty by default; set
    // it in tests and in any system where a silently stale estimate is worse
    // than a loud log line.
    using ErrorHandler = std::function<void(FilterError)>;

    // Constructs a filter whose state and covariance must be set with Reset()
    // before use.
    ExtendedKalmanFilter();

    ExtendedKalmanFilter(const StateVector& initial_state,
                         const StateMatrix& initial_covariance);

    virtual ~ExtendedKalmanFilter() = default;

    // ---- configuration ---------------------------------------------------
    // When true (default) the covariance is symmetrized after every step.
    // Cheap, and it keeps covariance "drift" from turning into a non-PSD
    // matrix over long runs.
    bool symmetrize_covariance() const { return symmetrize_covariance_; }
    void set_symmetrize_covariance(bool enable) {
        symmetrize_covariance_ = enable;
    }

    void set_error_handler(ErrorHandler handler) {
        error_handler_ = std::move(handler);
    }

    // ---- accessors -------------------------------------------------------
    const StateVector& state() const { return x_; }
    const StateMatrix& covariance() const { return P_; }
    int state_dimension() const { return static_cast<int>(x_.size()); }

    // Innovation and its covariance from the last successful
    // Update()/Correct().
    const MeasurementVector& innovation() const { return innovation_; }
    const MeasurementMatrix& innovation_covariance() const {
        return innovation_covariance_;
    }
    // Kalman gain used by the last successful Update()/Correct().
    const Eigen::MatrixXd& kalman_gain() const { return kalman_gain_; }

    bool Reset(const StateVector& initial_state,
               const StateMatrix& initial_covariance);
    void set_state(const StateVector& state) { x_ = state; }
    void set_covariance(const StateMatrix& covariance);

    // ---- prediction ------------------------------------------------------
    // x^- = f(x), P^- = F P F^T + Q. `jacobian` may be an empty std::function,
    // in which case F is obtained by finite differences of `transition`.
    // Returns false (and leaves the filter untouched) on non-finite input or
    // a diverging model output.
    bool Predict(
        const StateTransitionFunction& transition,
        const StateMatrix& process_noise,
        const StateTransitionJacobian& jacobian = StateTransitionJacobian());
    bool Predict(const StateTransitionFunction& transition,
                 const StateTransitionJacobian& jacobian,
                 const StateMatrix& process_noise) {
        return Predict(transition, process_noise, jacobian);
    }

    // ---- measurement update ---------------------------------------------
    // Returns false (leaving the predicted state in place) when the input is
    // unusable: wrong dimensions, non-finite values, singular innovation
    // covariance, or a non-PSD R.
    bool Update(const MeasurementFunction& measurement,
                const MeasurementVector& y,
                const StateMatrix& measurement_noise,
                const MeasurementJacobian& jacobian = MeasurementJacobian());
    bool Update(const MeasurementFunction& measurement,
                const MeasurementJacobian& jacobian, const MeasurementVector& y,
                const StateMatrix& measurement_noise) {
        return Update(measurement, y, measurement_noise, jacobian);
    }

    // ---- one-call convenience API ---------------------------------------
    // "Correct" mirrors the classic Kalman naming (predict + correct).
    bool Correct(const MeasurementFunction& measurement,
                 const MeasurementVector& y,
                 const StateMatrix& measurement_noise,
                 const MeasurementJacobian& jacobian = MeasurementJacobian()) {
        return Update(measurement, y, measurement_noise, jacobian);
    }

 protected:
    // Central finite-difference Jacobian of a vector function.
    StateMatrix NumericalJacobian(const StateVector& x,
                                  const StateTransitionFunction& function);
    MeasurementMatrix NumericalMeasurementJacobian(
        const StateVector& x, const MeasurementFunction& function);

    // Shared post-processing of a state/covariance pair after a step.
    // Returns false when the pair is unusable.
    bool FinalizeStep(const StateVector& state, const StateMatrix& covariance);

    // Reports a refusal to the error handler, if one is installed.
    void ReportError(FilterError error) const {
        if (error_handler_) {
            error_handler_(error);
        }
    }

    StateVector x_;
    StateMatrix P_;
    MeasurementVector innovation_;
    MeasurementMatrix innovation_covariance_;
    Eigen::MatrixXd kalman_gain_;
    ErrorHandler error_handler_;
    bool symmetrize_covariance_ = true;
};

// Iterated extended Kalman filter (IEKF).
//
// The plain EKF linearizes h once, at the predicted state x^-. IEKF re-solves
// the measurement update with a Gauss-Newton iteration, re-linearizing h at
// each iterate, which removes most of the linearization error of the update
// and is noticeably better when the measurement is strongly nonlinear (range
// or bearing sensors, polar-to-Cartesian conversions, ...).
//
// This implementation follows the standard "prior covariance" form: the gain
// is computed from the predicted covariance P^- at every iteration, so the
// result is a damped Gauss-Newton step and never overshoots as badly as the
// naive form that uses the already-updated covariance.
class IteratedExtendedKalmanFilter : public ExtendedKalmanFilter {
 public:
    IteratedExtendedKalmanFilter() = default;
    IteratedExtendedKalmanFilter(const StateVector& initial_state,
                                 const StateMatrix& initial_covariance)
        : ExtendedKalmanFilter(initial_state, initial_covariance) {}

    int max_iterations() const { return max_iterations_; }
    void set_max_iterations(int iterations) {
        max_iterations_ = iterations > 1 ? iterations : 1;
    }
    // Gauss-Newton stops early once the state move is below this relative
    // threshold.
    double convergence_tolerance() const { return convergence_tolerance_; }
    void set_convergence_tolerance(double tolerance) {
        convergence_tolerance_ = tolerance > 0.0 ? tolerance : 1e-12;
    }
    int last_iterations() const { return last_iterations_; }

    // Same contract as ExtendedKalmanFilter::Update, but iterated. When
    // max_iterations() == 1 this is exactly the EKF update.
    bool UpdateIterated(
        const MeasurementFunction& measurement, const MeasurementVector& y,
        const StateMatrix& measurement_noise,
        const MeasurementJacobian& jacobian = MeasurementJacobian());

 private:
    int max_iterations_ = 5;
    double convergence_tolerance_ = 1e-9;
    int last_iterations_ = 0;
};

}  // namespace kalman

#endif  // KALMAN_EXTENDED_KALMAN_FILTER_H_
