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

#include "kalman/unscented_kalman_filter.h"

#include <cmath>

namespace kalman {

UnscentedKalmanFilter::UnscentedKalmanFilter()
    : x_(StateVector()),
      P_(StateMatrix()),
      innovation_(MeasurementVector()),
      innovation_covariance_(MeasurementMatrix()) {}

UnscentedKalmanFilter::UnscentedKalmanFilter(
    const StateVector& initial_state, const StateMatrix& initial_covariance)
    : UnscentedKalmanFilter() {
    Reset(initial_state, initial_covariance);
}

void UnscentedKalmanFilter::set_alpha(double alpha) {
    if (alpha > 0.0 && std::isfinite(alpha)) {
        alpha_ = alpha;
    }
}

bool UnscentedKalmanFilter::Reset(const StateVector& initial_state,
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
    UpdateWeights();
    return true;
}

void UnscentedKalmanFilter::set_covariance(const StateMatrix& covariance) {
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

void UnscentedKalmanFilter::UpdateWeights() {
    const int n = static_cast<int>(x_.size());
    if (n <= 0) {
        return;
    }
    const double kappa =
        kappa_is_default_ ? 3.0 - static_cast<double>(n) : kappa_;
    const double lambda = alpha_ * alpha_ * (static_cast<double>(n) + kappa) -
                          static_cast<double>(n);
    const double denominator = static_cast<double>(n) + lambda;
    if (!(denominator > 0.0)) {
        return;
    }

    weights_mean_ =
        Eigen::VectorXd::Constant(2 * n + 1, 1.0 / (2.0 * denominator));
    weights_covariance_ = weights_mean_;
    weights_mean_(0) = lambda / denominator;
    weights_covariance_(0) =
        lambda / denominator + (1.0 - alpha_ * alpha_ + beta_);
}

bool UnscentedKalmanFilter::GenerateSigmaPoints(const StateVector& mean,
                                                const StateMatrix& covariance,
                                                Eigen::MatrixXd* sigma_points) {
    const int n = static_cast<int>(mean.size());
    if (sigma_points == nullptr || covariance.rows() != n ||
        covariance.cols() != n || !AllFinite(mean) || !AllFinite(covariance)) {
        return false;
    }
    const double kappa =
        kappa_is_default_ ? 3.0 - static_cast<double>(n) : kappa_;
    const double lambda = alpha_ * alpha_ * (static_cast<double>(n) + kappa) -
                          static_cast<double>(n);
    const double scale = static_cast<double>(n) + lambda;
    if (!(scale > 0.0)) {
        return false;
    }

    StateMatrix scaled = covariance * scale;
    if (!RegularizeInPlace(scaled)) {
        return false;
    }
    const Eigen::LLT<StateMatrix> llt(scaled);
    if (llt.info() != Eigen::Success) {
        return false;
    }
    const StateMatrix root = llt.matrixL();

    sigma_points->resize(n, 2 * n + 1);
    sigma_points->col(0) = mean;
    for (int i = 0; i < n; ++i) {
        sigma_points->col(i + 1) = mean + root.col(i);
        sigma_points->col(i + 1 + n) = mean - root.col(i);
    }
    return AllFinite(*sigma_points);
}

bool UnscentedKalmanFilter::Predict(const StateTransitionFunction& transition,
                                    const StateMatrix& process_noise) {
    const int n = static_cast<int>(x_.size());
    if (n <= 0 || !transition) {
        return false;
    }
    if (process_noise.rows() != n || process_noise.cols() != n) {
        return false;
    }
    if (!AllFinite(process_noise) || !IsPositiveSemidefinite(process_noise)) {
        return false;
    }
    UpdateWeights();
    if (weights_mean_.size() != 2 * n + 1) {
        return false;
    }

    Eigen::MatrixXd sigma_points;
    if (!GenerateSigmaPoints(x_, P_, &sigma_points)) {
        return false;
    }

    Eigen::MatrixXd propagated(n, 2 * n + 1);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const StateVector point = transition(sigma_points.col(i));
        if (point.size() != n || !AllFinite(point)) {
            return false;
        }
        propagated.col(i) = point;
    }

    const StateVector predicted_mean = propagated * weights_mean_;
    if (!AllFinite(predicted_mean)) {
        return false;
    }
    StateMatrix predicted_covariance = process_noise;
    for (int i = 0; i < 2 * n + 1; ++i) {
        const StateVector deviation = propagated.col(i) - predicted_mean;
        predicted_covariance.noalias() +=
            weights_covariance_(i) * deviation * deviation.transpose();
    }
    if (!AllFinite(predicted_covariance) ||
        !IsPositiveSemidefinite(predicted_covariance)) {
        return false;
    }
    SymmetrizeInPlace(predicted_covariance);

    x_ = predicted_mean;
    P_ = predicted_covariance;
    return true;
}

bool UnscentedKalmanFilter::PredictLinear(const StateMatrix& state_transition,
                                          const StateMatrix& process_noise) {
    const StateMatrix a = state_transition;
    if (a.rows() != x_.size() || a.cols() != x_.size()) {
        return false;
    }
    return Predict([&a](const StateVector& x) -> StateVector { return a * x; },
                   process_noise);
}

bool UnscentedKalmanFilter::Correct(const MeasurementFunction& measurement,
                                    const MeasurementVector& y,
                                    const StateMatrix& measurement_noise) {
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
    UpdateWeights();
    if (weights_mean_.size() != 2 * n + 1) {
        return false;
    }

    Eigen::MatrixXd sigma_points;
    if (!GenerateSigmaPoints(x_, P_, &sigma_points)) {
        return false;
    }

    Eigen::MatrixXd measurements(m, 2 * n + 1);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const MeasurementVector point = measurement(sigma_points.col(i));
        if (point.size() != m || !AllFinite(point)) {
            return false;
        }
        measurements.col(i) = point;
    }

    const MeasurementVector predicted_measurement =
        measurements * weights_mean_;
    if (!AllFinite(predicted_measurement)) {
        return false;
    }

    MeasurementMatrix innovation_covariance = measurement_noise;
    Eigen::MatrixXd cross_covariance = Eigen::MatrixXd::Zero(n, m);
    for (int i = 0; i < 2 * n + 1; ++i) {
        const StateVector dx = sigma_points.col(i) - x_;
        const MeasurementVector dy =
            measurements.col(i) - predicted_measurement;
        innovation_covariance.noalias() +=
            weights_covariance_(i) * dy * dy.transpose();
        cross_covariance.noalias() +=
            weights_covariance_(i) * dx * dy.transpose();
    }
    SymmetrizeInPlace(innovation_covariance);
    const Eigen::MatrixXd gain =
        SolveKalmanGain(innovation_covariance, cross_covariance);
    if (gain.rows() != n || gain.cols() != m) {
        return false;
    }

    const MeasurementVector innovation = y - predicted_measurement;
    const StateVector updated_mean = x_ + gain * innovation;
    const StateMatrix updated_covariance =
        P_ - gain * innovation_covariance * gain.transpose();
    if (!AllFinite(updated_mean) || !AllFinite(updated_covariance) ||
        !IsPositiveSemidefinite(updated_covariance)) {
        return false;
    }

    x_ = updated_mean;
    P_ = updated_covariance;
    SymmetrizeInPlace(P_);
    innovation_ = innovation;
    innovation_covariance_ = innovation_covariance;
    kalman_gain_ = gain;
    return true;
}

bool UnscentedKalmanFilter::CorrectLinear(
    const MeasurementMatrix& measurement_matrix, const MeasurementVector& y,
    const StateMatrix& measurement_noise) {
    const MeasurementMatrix c = measurement_matrix;
    if (c.cols() != x_.size() || c.rows() != y.size()) {
        return false;
    }
    return Correct(
        [&c](const StateVector& x) -> MeasurementVector { return c * x; }, y,
        measurement_noise);
}

}  // namespace kalman
