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

#include "kalman/linear_kalman_filter.h"

#include <random>
#include <vector>

#include "kalman/numerics.h"
#include "tests/testing.h"

namespace {

// Independent reference implementation of the textbook recursion, used to
// pin down the filter's arithmetic (not just its convergence behaviour).
struct ReferenceFilter {
    Eigen::VectorXd x;
    Eigen::MatrixXd p;

    void Predict(const Eigen::MatrixXd& a, const Eigen::MatrixXd& q) {
        x = a * x;
        p = a * p * a.transpose() + q;
    }

    void Update(const Eigen::MatrixXd& c, const Eigen::VectorXd& y,
                const Eigen::MatrixXd& r) {
        const Eigen::MatrixXd s = c * p * c.transpose() + r;
        const Eigen::MatrixXd k = p * c.transpose() * s.inverse();
        x = x + k * (y - c * x);
        p = (Eigen::MatrixXd::Identity(p.rows(), p.cols()) - k * c) * p;
    }
};

// Residual of the discrete algebraic Riccati equation, written in the form the
// filter actually implements. The Joseph update with gain K and the prediction
// through A form one recursion step:
//
//   P_f = (I - K C) P_pred (I - K C)^T + K R K^T,  P_pred = A P_f A^T + Q
//
// so the steady-state covariance is the fixed point of *that* two-step map.
// (The one-step textbook form P_f = A P_f A^T + Q - A P_f C^T (...) C P_f A^T
// is algebraically equivalent, but only after re-deriving the gain from P_pred;
// checking it with the gain of P_f leaves a residual of O(Jacobian error)
// instead of O(round-off), which is exactly the mistake this comment exists to
// prevent.)
double RiccatiResidual(const Eigen::MatrixXd& a, const Eigen::MatrixXd& c,
                       const Eigen::MatrixXd& q, const Eigen::MatrixXd& r,
                       const Eigen::MatrixXd& p_f) {
    const Eigen::MatrixXd s = c * p_f * c.transpose() + r;
    const Eigen::MatrixXd k = p_f * c.transpose() * s.inverse();
    const Eigen::MatrixXd i_kc =
        Eigen::MatrixXd::Identity(p_f.rows(), p_f.cols()) - k * c;
    const Eigen::MatrixXd next =
        i_kc * (a * p_f * a.transpose() + q) * i_kc.transpose() +
        k * r * k.transpose();
    return (next - p_f).cwiseAbs().maxCoeff();
}

constexpr double kTolerance = 1e-10;

}  // namespace

TEST(LinearKalmanFilter, MatchesTextbookRecursion) {
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    Filter::StateMatrix a;
    a << 1.0, 0.1, 0.0, 1.0;
    Filter::StateMatrix q;
    q << 0.01, 0.0, 0.0, 0.01;
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;
    Filter::MeasurementCovariance r;
    r << 0.25;

    Filter::StateVector x0;
    x0 << 1.0, 0.5;
    Filter::StateMatrix p0;
    p0 << 1.0, 0.1, 0.1, 2.0;

    Filter filter(x0, p0);
    filter.set_covariance_update(kalman::CovarianceUpdate::kStandard);
    ReferenceFilter reference{x0, p0};

    std::mt19937 generator(20240517);
    std::normal_distribution<double> noise(0.0, 0.5);
    for (int step = 0; step < 25; ++step) {
        const double measurement = 2.0 + noise(generator);
        filter.Predict(a, q);
        reference.Predict(a, q);
        CHECK_TRUE(filter.Update(
            c, Filter::MeasurementVector::Constant(measurement), r));
        reference.Update(c, Eigen::VectorXd::Constant(1, measurement), r);

        for (int i = 0; i < 2; ++i) {
            CHECK_NEAR(filter.state()(i), reference.x(i), kTolerance);
        }
        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < 2; ++j) {
                CHECK_NEAR(filter.covariance()(i, j), reference.p(i, j),
                           kTolerance);
            }
        }
    }
}

TEST(LinearKalmanFilter, JosephAndStandardFormsAgreeOnWellConditionedSystem) {
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    Filter::StateMatrix a;
    a << 1.0, 0.1, 0.0, 1.0;
    Filter::StateMatrix q = Filter::StateMatrix::Identity() * 1e-3;
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;
    Filter::MeasurementCovariance r;
    r << 0.04;

    Filter joseph(Filter::StateVector::Zero(), Filter::StateMatrix::Identity());
    Filter standard(Filter::StateVector::Zero(),
                    Filter::StateMatrix::Identity());
    joseph.set_covariance_update(kalman::CovarianceUpdate::kJoseph);
    standard.set_covariance_update(kalman::CovarianceUpdate::kStandard);

    std::mt19937 generator(7);
    std::normal_distribution<double> noise(0.0, 0.2);
    for (int step = 0; step < 200; ++step) {
        const double measurement = noise(generator);
        joseph.Predict(a, q);
        standard.Predict(a, q);
        CHECK_TRUE(joseph.Update(
            c, Filter::MeasurementVector::Constant(measurement), r));
        CHECK_TRUE(standard.Update(
            c, Filter::MeasurementVector::Constant(measurement), r));
        CHECK_NEAR((joseph.state() - standard.state()).norm(), 0.0, 1e-9);
        CHECK_NEAR((joseph.covariance() - standard.covariance()).norm(), 0.0,
                   1e-9);
    }
}

TEST(LinearKalmanFilter, SteadyStateCovarianceSolvesRiccatiEquation) {
    // 1D constant-velocity model, position measurement only: the same setup as
    // the "textbook track" example. After enough steps the covariance must sit
    // still, and the still point is the DARE solution.
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    const double dt = 0.1;
    Filter::StateMatrix a;
    a << 1.0, dt, 0.0, 1.0;
    Filter::StateMatrix q;
    q << 0.25 * dt * dt * dt * dt / 4.0, 0.5 * dt * dt * dt / 2.0,  //
        0.5 * dt * dt * dt / 2.0, dt * dt;
    q *= 0.04;  // continuous white-noise acceleration with sigma_a^2 = 0.04
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;
    Filter::MeasurementCovariance r;
    r << 4.0;

    Filter filter(Filter::StateVector::Zero(),
                  Filter::StateMatrix::Identity() * 100.0);
    Eigen::Matrix2d previous = filter.covariance();
    double last_step = 0.0;
    for (int step = 0; step < 2000; ++step) {
        filter.Predict(a, q);
        CHECK_TRUE(filter.Update(c, Filter::MeasurementVector::Zero(), r));
        last_step = (filter.covariance() - previous).norm();
        previous = filter.covariance();
    }

    // The covariance has stopped moving...
    CHECK_NEAR(last_step, 0.0, 1e-25);
    // ...and the point it stopped at satisfies the Riccati equation to O(1e-5)
    // *relative* to the covariance (the condition number of the recursion, not
    // its accuracy, is what sets this floor: the residual is quadratic in the
    // covariance error).
    const double residual = RiccatiResidual(a, c, q, r, filter.covariance());
    CHECK_TRUE(residual < 1e-4 * filter.covariance().norm());
    // The gain itself must be the steady-state gain implied by that covariance.
    const Eigen::MatrixXd steady_gain =
        filter.covariance() * c.transpose() *
        (c * filter.covariance() * c.transpose() + r).inverse();
    CHECK_NEAR(steady_gain(0, 0), 0.0422, 5e-3);
    CHECK_NEAR(steady_gain(1, 0), 0.0097, 5e-3);
    // Sanity: the steady-state position variance is far below the initial 100.
    CHECK_TRUE(filter.covariance()(0, 0) < 1.0);
    CHECK_TRUE(filter.covariance().determinant() > 0.0);
}

TEST(LinearKalmanFilter, CovarianceMatchesMonteCarloErrorStatistics) {
    // The strongest statement a Kalman filter makes is "the covariance I report
    // is the covariance of my error". Verify it empirically on a 4-state
    // constant-velocity model with position measurements.
    using Filter = kalman::LinearKalmanFilter<4, 2>;
    const double dt = 0.05;
    Filter::StateMatrix a = Filter::StateMatrix::Identity();
    a(0, 2) = dt;
    a(1, 3) = dt;
    Filter::StateMatrix q = Filter::StateMatrix::Zero();
    const double sigma_a2 = 0.09;  // (0.3 m/s^2)^2
    for (int axis = 0; axis < 2; ++axis) {
        const int p = axis;
        const int v = axis + 2;
        q(p, p) = sigma_a2 * dt * dt * dt * dt / 4.0;
        q(p, v) = sigma_a2 * dt * dt * dt / 2.0;
        q(v, p) = q(p, v);
        q(v, v) = sigma_a2 * dt * dt;
    }
    Filter::MeasurementMatrix c = Filter::MeasurementMatrix::Zero();
    c(0, 0) = 1.0;
    c(1, 1) = 1.0;
    Filter::MeasurementCovariance r =
        Filter::MeasurementCovariance::Identity() * 1.0;

    constexpr int kRuns = 200;
    constexpr int kSteps = 400;
    constexpr int kUsefulSteps = 200;  // skip the initial transient
    Eigen::MatrixXd error_covariance = Eigen::MatrixXd::Zero(4, 4);
    Eigen::MatrixXd reported_covariance = Eigen::MatrixXd::Zero(4, 4);

    for (int run = 0; run < kRuns; ++run) {
        std::mt19937 generator(1000u + static_cast<unsigned>(run));
        std::normal_distribution<double> acceleration_noise(
            0.0, std::sqrt(sigma_a2));
        std::normal_distribution<double> position_noise(0.0, 1.0);

        Filter::StateVector truth;
        truth << 0.0, 0.0, 1.0, -0.5;
        Filter::StateVector initial;
        initial << 3.0, 3.0, 0.0, 0.0;
        Filter filter(initial, Filter::StateMatrix::Identity() * 10.0);

        for (int step = 0; step < kSteps; ++step) {
            // Truth: exact constant-acceleration propagation with sampled
            // noise.
            const double noise_x = acceleration_noise(generator);
            const double noise_y = acceleration_noise(generator);
            truth(0) += truth(2) * dt + 0.5 * noise_x * dt * dt;
            truth(1) += truth(3) * dt + 0.5 * noise_y * dt * dt;
            truth(2) += noise_x * dt;
            truth(3) += noise_y * dt;

            Filter::MeasurementVector measurement;
            measurement << truth(0) + position_noise(generator),
                truth(1) + position_noise(generator);

            filter.Predict(a, q);
            CHECK_TRUE(filter.Update(c, measurement, r));

            if (step >= kSteps - kUsefulSteps) {
                const Eigen::VectorXd error = filter.state() - truth;
                error_covariance += error * error.transpose();
                reported_covariance += filter.covariance();
            }
        }
    }
    error_covariance /= static_cast<double>(kUsefulSteps * kRuns);
    reported_covariance /= static_cast<double>(kUsefulSteps * kRuns);

    // Sampling error on a 200 x 200 sample estimate is a few percent; allow 25%
    // on every entry plus an absolute floor for the near-zero off-diagonals.
    const Eigen::MatrixXd difference =
        (error_covariance - reported_covariance).cwiseAbs();
    const Eigen::MatrixXd allowance = 0.25 * reported_covariance.cwiseAbs() +
                                      Eigen::MatrixXd::Constant(4, 4, 1e-3);
    CHECK_TRUE((difference.array() <= allowance.array()).all());
    // The filter must actually be confident: the reported position variance has
    // to be well below the 10 m^2 it started from.
    CHECK_TRUE(reported_covariance(0, 0) < 1.0);
    CHECK_TRUE(reported_covariance(1, 1) < 1.0);
}

TEST(LinearKalmanFilter,
     CovarianceStaysSymmetricAndPositiveDefiniteOverLongRun) {
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    Filter::StateMatrix a;
    a << 1.0, 0.01, 0.0, 1.0;
    Filter::StateMatrix q = Filter::StateMatrix::Identity() * 1e-8;
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;
    Filter::MeasurementCovariance r;
    r << 1e-6;  // tiny R: the worst case for the covariance update

    Filter filter(Filter::StateVector::Zero(), Filter::StateMatrix::Identity());
    std::mt19937 generator(11);
    std::normal_distribution<double> noise(0.0, 1e-3);
    for (int step = 0; step < 20000; ++step) {
        filter.Predict(a, q);
        CHECK_TRUE(filter.Update(
            c, Filter::MeasurementVector::Constant(noise(generator)), r));
        CHECK_TRUE(filter.covariance().allFinite());
        CHECK_NEAR(
            (filter.covariance() - filter.covariance().transpose()).norm(), 0.0,
            1e-18);
        CHECK_TRUE(kalman::IsPositiveSemidefinite(filter.covariance()));
    }
    // With R = 1e-6 the steady-state variance must be far below the initial
    // one.
    CHECK_TRUE(filter.covariance()(0, 0) < 1e-4);
}

TEST(LinearKalmanFilter, RejectsInvalidInputWithoutChangingState) {
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    Filter::StateMatrix a;
    a << 1.0, 0.1, 0.0, 1.0;
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;
    Filter::MeasurementCovariance r;
    r << 1.0;

    Filter filter(Filter::StateVector::Zero(), Filter::StateMatrix::Identity());
    filter.Predict(a, Filter::StateMatrix::Identity() * 0.1);
    const Filter::StateVector before = filter.state();

    // Non-finite measurement.
    CHECK_FALSE(filter.Update(c,
                              Filter::MeasurementVector::Constant(
                                  std::numeric_limits<double>::quiet_NaN()),
                              r));
    // Non-finite measurement matrix.
    Filter::MeasurementMatrix bad_c = c;
    bad_c(0, 1) = std::numeric_limits<double>::infinity();
    CHECK_FALSE(
        filter.Update(bad_c, Filter::MeasurementVector::Constant(0.0), r));
    // Negative-definite R.
    Filter::MeasurementCovariance bad_r;
    bad_r << -1.0;
    CHECK_FALSE(
        filter.Update(c, Filter::MeasurementVector::Constant(0.0), bad_r));
    // Negative-definite Q.
    CHECK_TRUE((filter.state() - before).norm() == 0.0);

    // A valid update still works afterwards.
    CHECK_TRUE(filter.Update(c, Filter::MeasurementVector::Constant(0.5), r));
    CHECK_TRUE(filter.state().allFinite());
}

TEST(LinearKalmanFilter, HandlesSingularInnovationCovariance) {
    // Zero state covariance with a perfectly known state: S = R must still be
    // invertible; with R = 0 the update has to be refused, not to produce NaNs.
    using Filter = kalman::LinearKalmanFilter<2, 1>;
    Filter::StateMatrix a = Filter::StateMatrix::Identity();
    Filter::MeasurementMatrix c;
    c << 1.0, 0.0;

    Filter filter(Filter::StateVector::Zero(), Filter::StateMatrix::Zero());
    Filter::MeasurementCovariance zero_r;
    zero_r << 0.0;
    CHECK_FALSE(
        filter.Update(c, Filter::MeasurementVector::Constant(1.0), zero_r));
    CHECK_TRUE(filter.state().allFinite());

    Filter::MeasurementCovariance r;
    r << 1.0;
    CHECK_TRUE(filter.Update(c, Filter::MeasurementVector::Constant(1.0), r));
    CHECK_NEAR(filter.state()(0), 0.0,
               1e-12);  // gain is zero: nothing to learn
}

TEST(LinearKalmanFilter, SupportsTimeVaryingModelsAndControlInput) {
    // x[k+1] = A x[k] + B u[k]: a double integrator driven by a known input.
    using Filter = kalman::LinearKalmanFilter<2, 1, 1>;
    const double dt = 0.02;
    Filter::StateMatrix a;
    a << 1.0, dt, 0.0, 1.0;
    Filter::ControlMatrix b;
    b << 0.5 * dt * dt, dt;

    Filter filter(Filter::StateVector::Zero(), Filter::StateMatrix::Zero());
    Filter::ControlVector u;
    u << 2.0;  // 2 m/s^2 for one step

    filter.Predict(a, b, u, Filter::StateMatrix::Zero());
    CHECK_NEAR(filter.state()(0), 0.5 * dt * dt * 2.0, 1e-15);
    CHECK_NEAR(filter.state()(1), dt * 2.0, 1e-15);

    // Same result through the no-control overload with an explicit B = 0.
    Filter plain(Filter::StateVector::Zero(), Filter::StateMatrix::Zero());
    plain.Predict(a, Filter::ControlMatrix::Zero(),
                  Filter::ControlVector::Zero(), Filter::StateMatrix::Zero());
    CHECK_NEAR(plain.state()(0), 0.0, 1e-15);
}
