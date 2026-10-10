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

#include "kalman/extended_kalman_filter.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace kalman {
namespace {

// Relative step used by the finite-difference Jacobians. 1e-6 is the sweet
// spot for double precision: truncation error ~ h^2 and rounding error ~ 1/h
// are balanced at h ~ 1e-5..1e-6.
constexpr double kFiniteDifferenceStep = 1e-6;

}  // namespace

const char* ToString(FilterError error) {
    switch (error) {
        case FilterError::kNone:
            return "no error";
        case FilterError::kInvalidDimensions:
            return "input has the wrong dimensions";
        case FilterError::kNonFiniteInput:
            return "input contains NaN or infinity";
        case FilterError::kCovarianceNotPositiveSemidefinite:
            return "covariance is not positive semidefinite";
        case FilterError::kSingularInnovationCovariance:
            return "innovation covariance is singular";
        case FilterError::kNonFiniteModelOutput:
            return "model function returned NaN or infinity";
        case FilterError::kJacobianDimensionMismatch:
            return "Jacobian has the wrong shape for this filter";
        case FilterError::kNonFiniteJacobian:
            return "Jacobian contains NaN or infinity";
    }
    return "unknown error";
}

ExtendedKalmanFilter::ExtendedKalmanFilter()
    : x_(StateVector()),
      P_(StateMatrix()),
      innovation_(MeasurementVector()),
      innovation_covariance_(MeasurementMatrix()) {}

ExtendedKalmanFilter::ExtendedKalmanFilter(
    const StateVector& initial_state, const StateMatrix& initial_covariance)
    : ExtendedKalmanFilter() {
    Reset(initial_state, initial_covariance);
}

bool ExtendedKalmanFilter::Reset(const StateVector& initial_state,
                                 const StateMatrix& initial_covariance) {
    if (initial_state.size() <= 0 || !AllFinite(initial_state)) {
        return false;
    }
    StateMatrix candidate = initial_covariance;
    if (candidate.rows() != initial_state.size() ||
        candidate.cols() != initial_state.size()) {
        return false;
    }
    if (!AllFinite(candidate) || !IsPositiveSemidefinite(candidate)) {
        return false;
    }
    SymmetrizeInPlace(candidate);
    x_ = initial_state;
    P_ = candidate;
    return true;
}

void ExtendedKalmanFilter::set_covariance(const StateMatrix& covariance) {
    if (covariance.rows() != x_.size() || covariance.cols() != x_.size()) {
        return;
    }
    StateMatrix candidate = covariance;
    if (!AllFinite(candidate) || !IsPositiveSemidefinite(candidate)) {
        return;
    }
    SymmetrizeInPlace(candidate);
    P_ = candidate;
}

ExtendedKalmanFilter::StateMatrix ExtendedKalmanFilter::NumericalJacobian(
    const StateVector& x, const StateTransitionFunction& function) {
    const int n = static_cast<int>(x.size());
    StateMatrix jacobian(n, n);
    StateVector perturbed = x;
    for (int j = 0; j < n; ++j) {
        const double step =
            kFiniteDifferenceStep * std::max(1.0, std::abs(x(j)));
        perturbed(j) = x(j) + step;
        const StateVector forward = function(perturbed);
        perturbed(j) = x(j) - step;
        const StateVector backward = function(perturbed);
        perturbed(j) = x(j);
        if (forward.size() != x.size() || backward.size() != x.size()) {
            return StateMatrix();
        }
        jacobian.col(j) = (forward - backward) / (2.0 * step);
    }
    return jacobian;
}

ExtendedKalmanFilter::MeasurementMatrix
ExtendedKalmanFilter::NumericalMeasurementJacobian(
    const StateVector& x, const MeasurementFunction& function) {
    const int n = static_cast<int>(x.size());
    const MeasurementVector reference = function(x);
    const int m = static_cast<int>(reference.size());
    MeasurementMatrix jacobian(m, n);
    StateVector perturbed = x;
    for (int j = 0; j < n; ++j) {
        const double step =
            kFiniteDifferenceStep * std::max(1.0, std::abs(x(j)));
        perturbed(j) = x(j) + step;
        const MeasurementVector forward = function(perturbed);
        perturbed(j) = x(j) - step;
        const MeasurementVector backward = function(perturbed);
        perturbed(j) = x(j);
        if (forward.size() != m || backward.size() != m) {
            return MeasurementMatrix();
        }
        jacobian.col(j) = (forward - backward) / (2.0 * step);
    }
    return jacobian;
}

bool ExtendedKalmanFilter::FinalizeStep(const StateVector& state,
                                        const StateMatrix& covariance) {
    if (!AllFinite(state) || !AllFinite(covariance)) {
        return false;
    }
    x_ = state;
    P_ = covariance;
    if (symmetrize_covariance_) {
        SymmetrizeInPlace(P_);
    }
    return true;
}

bool ExtendedKalmanFilter::Predict(const StateTransitionFunction& transition,
                                   const StateMatrix& process_noise,
                                   const StateTransitionJacobian& jacobian) {
    const int n = static_cast<int>(x_.size());
    if (n <= 0 || !transition) {
        ReportError(FilterError::kInvalidDimensions);
        return false;
    }
    if (process_noise.rows() != n || process_noise.cols() != n) {
        ReportError(FilterError::kInvalidDimensions);
        return false;
    }
    if (!AllFinite(process_noise)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }
    if (!IsPositiveSemidefinite(process_noise)) {
        ReportError(FilterError::kCovarianceNotPositiveSemidefinite);
        return false;
    }

    StateMatrix jacobian_eval(n, n);
    if (jacobian) {
        jacobian_eval = jacobian(x_);
        // The shape check is the only defence against passing the (identical
        // typed) measurement Jacobian here, so report it distinctly.
        if (jacobian_eval.rows() != n || jacobian_eval.cols() != n) {
            ReportError(FilterError::kJacobianDimensionMismatch);
            return false;
        }
        if (!AllFinite(jacobian_eval)) {
            ReportError(FilterError::kNonFiniteJacobian);
            return false;
        }
    } else {
        jacobian_eval = NumericalJacobian(x_, transition);
        if (jacobian_eval.rows() != n || jacobian_eval.cols() != n) {
            ReportError(FilterError::kNonFiniteModelOutput);
            return false;
        }
        if (!AllFinite(jacobian_eval)) {
            ReportError(FilterError::kNonFiniteModelOutput);
            return false;
        }
    }

    const StateVector predicted = transition(x_);
    if (predicted.size() != n || !AllFinite(predicted)) {
        ReportError(FilterError::kNonFiniteModelOutput);
        return false;
    }
    const StateMatrix predicted_covariance =
        jacobian_eval * P_ * jacobian_eval.transpose() + process_noise;
    if (!FinalizeStep(predicted, predicted_covariance)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }
    return true;
}

bool ExtendedKalmanFilter::Update(const MeasurementFunction& measurement,
                                  const MeasurementVector& y,
                                  const StateMatrix& measurement_noise,
                                  const MeasurementJacobian& jacobian) {
    const int n = static_cast<int>(x_.size());
    if (n <= 0 || !measurement) {
        ReportError(FilterError::kInvalidDimensions);
        return false;
    }
    const int m = static_cast<int>(y.size());
    if (m <= 0) {
        ReportError(FilterError::kInvalidDimensions);
        return false;
    }
    if (!AllFinite(y)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }
    if (measurement_noise.rows() != m || measurement_noise.cols() != m) {
        ReportError(FilterError::kInvalidDimensions);
        return false;
    }
    if (!AllFinite(measurement_noise)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }
    if (!IsPositiveSemidefinite(measurement_noise)) {
        ReportError(FilterError::kCovarianceNotPositiveSemidefinite);
        return false;
    }

    MeasurementMatrix jacobian_eval(m, n);
    if (jacobian) {
        jacobian_eval = jacobian(x_);
        if (jacobian_eval.rows() != m || jacobian_eval.cols() != n) {
            ReportError(FilterError::kJacobianDimensionMismatch);
            return false;
        }
        if (!AllFinite(jacobian_eval)) {
            ReportError(FilterError::kNonFiniteJacobian);
            return false;
        }
    } else {
        jacobian_eval = NumericalMeasurementJacobian(x_, measurement);
        if (jacobian_eval.rows() != m || jacobian_eval.cols() != n) {
            ReportError(FilterError::kNonFiniteModelOutput);
            return false;
        }
        if (!AllFinite(jacobian_eval)) {
            ReportError(FilterError::kNonFiniteModelOutput);
            return false;
        }
    }

    const MeasurementVector predicted_measurement = measurement(x_);
    if (predicted_measurement.size() != m ||
        !AllFinite(predicted_measurement)) {
        ReportError(FilterError::kNonFiniteModelOutput);
        return false;
    }
    const MeasurementVector innovation = y - predicted_measurement;

    const MeasurementMatrix s =
        jacobian_eval * P_ * jacobian_eval.transpose() + measurement_noise;
    const Eigen::MatrixXd gain =
        SolveKalmanGain(s, P_ * jacobian_eval.transpose());
    if (gain.rows() != n || gain.cols() != m) {
        ReportError(FilterError::kSingularInnovationCovariance);
        return false;
    }
    if (!AllFinite(innovation)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }

    const StateVector updated = x_ + gain * innovation;
    const StateMatrix i_kh = StateMatrix::Identity(n, n) - gain * jacobian_eval;
    const StateMatrix updated_covariance =
        i_kh * P_ * i_kh.transpose() +
        gain * measurement_noise * gain.transpose();
    if (!FinalizeStep(updated, updated_covariance)) {
        ReportError(FilterError::kNonFiniteInput);
        return false;
    }

    innovation_ = innovation;
    innovation_covariance_ = s;
    kalman_gain_ = gain;
    return true;
}

bool IteratedExtendedKalmanFilter::UpdateIterated(
    const MeasurementFunction& measurement, const MeasurementVector& y,
    const StateMatrix& measurement_noise, const MeasurementJacobian& jacobian) {
    const int n = static_cast<int>(x_.size());
    if (n <= 0 || !measurement) {
        return false;
    }
    const int m = static_cast<int>(y.size());
    if (m <= 0 || !AllFinite(y)) {
        return false;
    }
    if (measurement_noise.rows() != m || measurement_noise.cols() != m) {
        return false;
    }
    if (!AllFinite(measurement_noise) ||
        !IsPositiveSemidefinite(measurement_noise)) {
        return false;
    }

    const StateVector prior = x_;
    const StateMatrix prior_covariance = P_;
    StateVector iterate = prior;
    MeasurementMatrix jacobian_eval(m, n);
    MeasurementMatrix s(m, m);
    Eigen::MatrixXd gain(n, m);
    MeasurementVector innovation(m);
    bool converged = false;

    last_iterations_ = 0;
    for (int iteration = 0; iteration < max_iterations_; ++iteration) {
        if (jacobian) {
            jacobian_eval = jacobian(iterate);
        } else {
            jacobian_eval = NumericalMeasurementJacobian(iterate, measurement);
        }
        if (jacobian_eval.rows() != m || jacobian_eval.cols() != n ||
            !AllFinite(jacobian_eval)) {
            return false;
        }

        const MeasurementVector predicted_measurement = measurement(iterate);
        if (predicted_measurement.size() != m ||
            !AllFinite(predicted_measurement)) {
            return false;
        }
        // Innovation linearized about the current iterate.
        const MeasurementVector residual =
            y - predicted_measurement + jacobian_eval * (iterate - prior);

        s = jacobian_eval * prior_covariance * jacobian_eval.transpose() +
            measurement_noise;
        gain = SolveKalmanGain(s, prior_covariance * jacobian_eval.transpose());
        if (gain.rows() != n || gain.cols() != m) {
            return false;
        }

        const StateVector next = prior + gain * residual;
        if (!AllFinite(next)) {
            return false;
        }
        const double move = (next - iterate).norm();
        const double scale = std::max(1.0, iterate.norm());
        iterate = next;
        ++last_iterations_;
        if (move <= convergence_tolerance_ * scale) {
            converged = true;
            break;
        }
    }
    (void)converged;  // A non-converged run still returns the best iterate.

    // Covariance at the final linearization point.
    if (jacobian) {
        jacobian_eval = jacobian(iterate);
    } else {
        jacobian_eval = NumericalMeasurementJacobian(iterate, measurement);
    }
    if (!AllFinite(jacobian_eval)) {
        return false;
    }
    const StateMatrix i_kh = StateMatrix::Identity(n, n) - gain * jacobian_eval;
    const StateMatrix updated_covariance =
        i_kh * prior_covariance * i_kh.transpose() +
        gain * measurement_noise * gain.transpose();
    if (!FinalizeStep(iterate, updated_covariance)) {
        return false;
    }

    const MeasurementVector predicted_measurement = measurement(x_);
    if (predicted_measurement.size() == m) {
        innovation = y - predicted_measurement;
    } else {
        innovation = MeasurementVector::Zero(m);
    }
    innovation_ = innovation;
    innovation_covariance_ = s;
    kalman_gain_ = gain;
    return true;
}

}  // namespace kalman
