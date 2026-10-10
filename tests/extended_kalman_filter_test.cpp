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

#include <cmath>
#include <limits>
#include <random>
#include <string>

#include "kalman/linear_kalman_filter.h"
#include "kalman/numerics.h"
#include "tests/testing.h"

namespace {

// ------------------------------------------------------------ pendulum model
// theta'' = -(g/l) sin(theta) - b theta',  y = theta
constexpr double kGravityOverLength = 10.0;
constexpr double kDamping = 0.2;
constexpr double kSampleTime = 0.05;
constexpr double kAngleSigma = 0.5 * M_PI / 180.0;
constexpr int kSubSteps = 20;

Eigen::Vector2d PendulumDynamics(const Eigen::Vector2d& state) {
    Eigen::Vector2d derivative;
    derivative << state(1),
        -kGravityOverLength * std::sin(state(0)) - kDamping * state(1);
    return derivative;
}

Eigen::Vector2d PendulumStep(const Eigen::Vector2d& state) {
    const double h = kSampleTime / kSubSteps;
    Eigen::Vector2d x = state;
    for (int i = 0; i < kSubSteps; ++i) {
        const Eigen::Vector2d k1 = PendulumDynamics(x);
        const Eigen::Vector2d k2 = PendulumDynamics(x + 0.5 * h * k1);
        const Eigen::Vector2d k3 = PendulumDynamics(x + 0.5 * h * k2);
        const Eigen::Vector2d k4 = PendulumDynamics(x + h * k3);
        x += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
    }
    kalman::NormalizeAnglesInPlace(x.head<1>());
    return x;
}

// exp(A_c Ts) for the 2x2 continuous-time Jacobian, via Cayley-Hamilton.
Eigen::Matrix2d PendulumDiscreteJacobian(const Eigen::Vector2d& state) {
    const double trace = -kDamping;
    const double determinant = kGravityOverLength * std::cos(state(0));
    const double discriminant = trace * trace - 4.0 * determinant;
    double c0 = 0.0;
    double c1 = 0.0;
    if (discriminant > 1e-12) {
        const double root = std::sqrt(discriminant);
        const double lambda1 = 0.5 * (trace + root);
        const double lambda2 = 0.5 * (trace - root);
        c1 = (std::exp(lambda1 * kSampleTime) -
              std::exp(lambda2 * kSampleTime)) /
             (lambda1 - lambda2);
        c0 = std::exp(lambda1 * kSampleTime) - c1 * lambda1;
    } else if (discriminant < -1e-12) {
        const double omega = 0.5 * std::sqrt(-discriminant);
        const double exponential = std::exp(0.5 * trace * kSampleTime);
        c1 = exponential * std::sin(omega * kSampleTime) / omega;
        c0 = exponential * std::cos(omega * kSampleTime) - c1 * 0.5 * trace;
    } else {
        const double half_trace = 0.5 * trace;
        const double exponential = std::exp(half_trace * kSampleTime);
        c0 = exponential * (1.0 - half_trace * kSampleTime);
        c1 = exponential * kSampleTime;
    }
    Eigen::Matrix2d continuous_jacobian;
    continuous_jacobian << 0.0, 1.0, -determinant, -kDamping;
    return c0 * Eigen::Matrix2d::Identity() + c1 * continuous_jacobian;
}

Eigen::VectorXd PendulumMeasurement(const Eigen::VectorXd& state) {
    Eigen::VectorXd measurement(1);
    measurement << state(0);
    return measurement;
}

Eigen::MatrixXd PendulumMeasurementJacobian(const Eigen::VectorXd& /*state*/) {
    Eigen::MatrixXd jacobian(1, 2);
    jacobian << 1.0, 0.0;
    return jacobian;
}

Eigen::VectorXd PendulumTransitionDynamic(const Eigen::VectorXd& state) {
    return PendulumStep(state);
}

// Hand-written Jacobian of the *transition* (df/dx, a 2x2 matrix). Note how
// easy it is to pass the 1x2 measurement Jacobian here instead - both callbacks
// have the same C++ type, so only the runtime shape check catches it. That is
// why ExtendedKalmanFilter reports the failure line through
// set_error_handler().
Eigen::MatrixXd PendulumTransitionJac(const Eigen::VectorXd& state) {
    return PendulumDiscreteJacobian(state);
}

const Eigen::Matrix2d& ProcessNoise() {
    static const Eigen::Matrix2d process_noise =
        (Eigen::Vector2d() << 1e-6, 1e-4).finished().asDiagonal();
    return process_noise;
}

Eigen::MatrixXd MeasurementNoise() {
    Eigen::MatrixXd noise(1, 1);
    noise << kAngleSigma * kAngleSigma;
    return noise;
}

struct PendulumErrors {
    double angle_rmse_deg = 0.0;
    double rate_rmse = 0.0;
};

// Runs the pendulum with a given transition-Jacobian callback (an empty
// std::function means "let the library use finite differences") and returns
// the steady-state RMSE of the two states.
PendulumErrors RunPendulum(
    const kalman::ExtendedKalmanFilter::StateTransitionJacobian& jacobian,
    unsigned int seed) {
    constexpr int kSteps = 400;
    constexpr int kSettlingSteps = 150;

    std::mt19937 generator(seed);
    std::normal_distribution<double> measurement_noise(0.0, kAngleSigma);
    std::normal_distribution<double> process_noise(0.0, 1.0);

    Eigen::VectorXd initial_state(2);
    initial_state << 30.0 * M_PI / 180.0, 0.0;
    Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Zero(2, 2);
    initial_covariance.diagonal() << 1e-6, 1e-4;

    kalman::ExtendedKalmanFilter filter(initial_state, initial_covariance);
    Eigen::Vector2d truth;
    truth << 30.0 * M_PI / 180.0, 0.0;

    double angle_square_sum = 0.0;
    double rate_square_sum = 0.0;
    int counted = 0;
    for (int step = 0; step < kSteps; ++step) {
        truth = PendulumStep(truth);
        truth(0) += 1e-3 * process_noise(generator);
        truth(1) += 1e-2 * process_noise(generator);

        Eigen::VectorXd measurement(1);
        measurement << truth(0) + measurement_noise(generator);

        CHECK_TRUE(filter.Predict(PendulumTransitionDynamic, ProcessNoise(),
                                  jacobian));
        CHECK_TRUE(filter.Update(PendulumMeasurement, measurement,
                                 MeasurementNoise(),
                                 PendulumMeasurementJacobian));

        if (step >= kSettlingSteps) {
            const double angle_error = kalman::AngularDifference(
                filter.state()(0), static_cast<double>(truth(0)));
            const double rate_error = filter.state()(1) - truth(1);
            angle_square_sum += angle_error * angle_error;
            rate_square_sum += rate_error * rate_error;
            ++counted;
        }
    }
    PendulumErrors errors;
    errors.angle_rmse_deg =
        std::sqrt(angle_square_sum / counted) * 180.0 / M_PI;
    errors.rate_rmse = std::sqrt(rate_square_sum / counted);
    return errors;
}

// ------------------------------------------------------- bearing-range model
// A target moving with constant velocity, observed from a fixed sensor that
// measures range and bearing: h(x) = [||p||, atan2(py, px)].
Eigen::VectorXd BearingRangeMeasurement(const Eigen::VectorXd& state) {
    Eigen::VectorXd measurement(2);
    measurement << std::hypot(state(0), state(1)),
        std::atan2(state(1), state(0));
    return measurement;
}

Eigen::MatrixXd BearingRangeJacobian(const Eigen::VectorXd& state) {
    const double px = state(0);
    const double py = state(1);
    const double squared_range = px * px + py * py;
    const double range = std::sqrt(squared_range);
    Eigen::MatrixXd jacobian(2, 4);
    jacobian << px / range, py / range, 0.0, 0.0,  //
        -py / squared_range, px / squared_range, 0.0, 0.0;
    return jacobian;
}

}  // namespace

TEST(ExtendedKalmanFilter, MatchesLinearKalmanFilterOnLinearSystem) {
    // With an affine model the EKF *is* the linear filter: the Jacobians are
    // constant, so every number must agree to machine precision. This is the
    // test that catches most implementation mistakes in the EKF recursion.
    constexpr double kDt = 0.1;
    Eigen::Matrix2d a;
    a << 1.0, kDt, 0.0, 1.0;
    const Eigen::Matrix2d q = Eigen::Matrix2d::Identity() * 1e-3;
    Eigen::Matrix<double, 1, 2> c;
    c << 1.0, 0.0;
    Eigen::Matrix<double, 1, 1> r;
    r << 0.25;

    Eigen::Vector2d initial;
    initial << 1.0, 0.0;
    const Eigen::Matrix2d initial_covariance = Eigen::Matrix2d::Identity();

    kalman::LinearKalmanFilter<2, 1> linear(initial, initial_covariance);
    kalman::ExtendedKalmanFilter extended(initial, initial_covariance);

    const kalman::ExtendedKalmanFilter::StateTransitionFunction transition =
        [&a](const Eigen::VectorXd& x) -> Eigen::VectorXd { return a * x; };
    const kalman::ExtendedKalmanFilter::StateTransitionJacobian
        transition_jacobian =
            [&a](const Eigen::VectorXd&) -> Eigen::MatrixXd { return a; };
    const kalman::ExtendedKalmanFilter::MeasurementFunction measurement =
        [&c](const Eigen::VectorXd& x) -> Eigen::VectorXd { return c * x; };

    std::mt19937 generator(12345);
    std::normal_distribution<double> noise(0.0, 0.5);
    for (int step = 0; step < 80; ++step) {
        const double value = noise(generator);
        linear.Predict(a, q);
        CHECK_TRUE(extended.Predict(transition, q, transition_jacobian));
        CHECK_TRUE(linear.Update(c, Eigen::VectorXd::Constant(1, value), r));
        CHECK_TRUE(extended.Update(
            measurement, Eigen::VectorXd::Constant(1, value), r,
            [&c](const Eigen::VectorXd&) -> Eigen::MatrixXd { return c; }));
        CHECK_NEAR((linear.state() - extended.state()).norm(), 0.0, 1e-12);
        CHECK_NEAR((linear.covariance() - extended.covariance()).norm(), 0.0,
                   1e-12);
    }

    // Same again with the Jacobians left out: the finite-difference machinery
    // must reproduce the analytic answer for a linear model.
    kalman::ExtendedKalmanFilter difference_filter(initial, initial_covariance);
    kalman::LinearKalmanFilter<2, 1> reference(initial, initial_covariance);
    for (int step = 0; step < 40; ++step) {
        const double value = noise(generator);
        difference_filter.Predict(transition, q);
        reference.Predict(a, q);
        difference_filter.Update(measurement,
                                 Eigen::VectorXd::Constant(1, value), r);
        reference.Update(c, Eigen::VectorXd::Constant(1, value), r);
        CHECK_NEAR((reference.state() - difference_filter.state()).norm(), 0.0,
                   1e-9);
        CHECK_NEAR(
            (reference.covariance() - difference_filter.covariance()).norm(),
            0.0, 1e-9);
    }
}

TEST(ExtendedKalmanFilter, FiniteDifferenceJacobianMatchesAnalytic) {
    // The library computes the transition Jacobian as a central difference of
    // the model function; reproduce that by hand and compare against the
    // closed-form exp(A_c Ts). The two differ only by the O(h^2) truncation
    // term of the difference, which must be tiny.
    Eigen::VectorXd state(2);
    state << 0.4, -0.3;
    const Eigen::MatrixXd analytic = PendulumDiscreteJacobian(state);

    // The true Jacobian of the discrete map is the solution of the variational
    // equation dPhi/dt = A_c(x(t)) Phi, Phi(0) = I, integrated alongside the
    // trajectory. That is what a central difference of the flow converges to,
    // so this is the honest reference to compare the filter against.
    const auto variational_jacobian = [](const Eigen::Vector2d& start) {
        constexpr int kFineSubSteps = 2000;
        const double h = kSampleTime / kFineSubSteps;
        Eigen::Vector2d x = start;
        Eigen::Matrix2d phi = Eigen::Matrix2d::Identity();
        const auto continuous_jacobian = [](const Eigen::Vector2d& state) {
            Eigen::Matrix2d jacobian;
            jacobian << 0.0, 1.0, -kGravityOverLength * std::cos(state(0)),
                -kDamping;
            return jacobian;
        };
        for (int i = 0; i < kFineSubSteps; ++i) {
            const Eigen::Vector2d k1 = PendulumDynamics(x);
            const Eigen::Matrix2d l1 = continuous_jacobian(x) * phi;
            const Eigen::Vector2d k2 = PendulumDynamics(x + 0.5 * h * k1);
            const Eigen::Matrix2d l2 =
                continuous_jacobian(x + 0.5 * h * k1) * (phi + 0.5 * h * l1);
            const Eigen::Vector2d k3 = PendulumDynamics(x + 0.5 * h * k2);
            const Eigen::Matrix2d l3 =
                continuous_jacobian(x + 0.5 * h * k2) * (phi + 0.5 * h * l2);
            const Eigen::Vector2d k4 = PendulumDynamics(x + h * k3);
            const Eigen::Matrix2d l4 =
                continuous_jacobian(x + h * k3) * (phi + h * l3);
            x += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            phi += h / 6.0 * (l1 + 2.0 * l2 + 2.0 * l3 + l4);
        }
        return phi;
    };

    Eigen::MatrixXd numerical(2, 2);
    for (int column = 0; column < 2; ++column) {
        const double step = 1e-6;
        Eigen::Vector2d forward = state;
        forward(column) += step;
        Eigen::Vector2d backward = state;
        backward(column) -= step;
        numerical.col(column) =
            (PendulumStep(forward) - PendulumStep(backward)) / (2.0 * step);
    }
    const Eigen::MatrixXd truth = variational_jacobian(state);
    CHECK_NEAR((numerical - truth).cwiseAbs().maxCoeff(), 0.0, 1e-8);

    // exp(A_c Ts) - the "freeze the Jacobian and exponentiate" shortcut - is
    // NOT the Jacobian of this discrete map: it drops the variation of A_c
    // along the trajectory and is wrong by ~1.7e-3 here (~7e-4 relative). Keep
    // that fact pinned down, because it is a tempting mistake.
    CHECK_TRUE((analytic - truth).cwiseAbs().maxCoeff() > 1e-4);

    // And the filter really uses that Jacobian: predicting with a rank-one
    // covariance must produce P = (F e_c)(F e_c)^T, i.e. the outer product of
    // the corresponding column of the Jacobian.
    kalman::ExtendedKalmanFilter filter(state, Eigen::MatrixXd::Identity(2, 2));
    for (int column = 0; column < 2; ++column) {
        Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(2, 2);
        covariance(column, column) = 1.0;
        CHECK_TRUE(filter.Reset(state, covariance));
        CHECK_TRUE(filter.Predict(PendulumTransitionDynamic,
                                  Eigen::MatrixXd::Zero(2, 2)));
        const Eigen::VectorXd expected = numerical.col(column);
        CHECK_NEAR(
            (filter.covariance() - expected * expected.transpose()).norm(), 0.0,
            1e-9);
    }
}

TEST(ExtendedKalmanFilter, TracksPendulumWithAngleOnlyMeasurement) {
    const PendulumErrors errors = RunPendulum(
        kalman::ExtendedKalmanFilter::StateTransitionJacobian(), 20240517);
    // Angle is measured, so it must be reconstructed to better than the sensor
    // noise; the rate is invisible and only the model constrains it.
    CHECK_TRUE(errors.angle_rmse_deg < 0.5);
    CHECK_TRUE(errors.rate_rmse < 0.05);

    // A frozen (never re-linearized) Jacobian is measurably worse: it keeps
    // propagating the covariance with the equilibrium dynamics.
    const PendulumErrors frozen = RunPendulum(
        [](const Eigen::VectorXd&) -> Eigen::MatrixXd {
            return PendulumDiscreteJacobian(
                (Eigen::Vector2d() << 0.0, 0.0).finished());
        },
        20240517);
    CHECK_TRUE(frozen.angle_rmse_deg > errors.angle_rmse_deg);
}

TEST(ExtendedKalmanFilter, RecoversSaneCovarianceForBearingRangeTracking) {
    // Constant-velocity target seen by a range/bearing sensor. The measurement
    // model is strongly nonlinear in the cross-range direction: the covariance
    // must stay positive definite and the estimate must stay consistent.
    constexpr double kDt = 0.1;
    Eigen::Matrix4d a = Eigen::Matrix4d::Identity();
    a(0, 2) = kDt;
    a(1, 3) = kDt;

    Eigen::Matrix4d q = Eigen::Matrix4d::Zero();
    const double acceleration_variance = 0.0025;  // (0.05 m/s^2)^2
    for (int axis = 0; axis < 2; ++axis) {
        const int p = axis;
        const int v = axis + 2;
        q(p, p) = acceleration_variance * std::pow(kDt, 4) / 4.0;
        q(p, v) = acceleration_variance * std::pow(kDt, 3) / 2.0;
        q(v, p) = q(p, v);
        q(v, v) = acceleration_variance * kDt * kDt;
    }

    Eigen::Matrix2d r = Eigen::Matrix2d::Zero();
    const double range_sigma = 3.0;
    const double bearing_sigma = 0.5 * M_PI / 180.0;
    r(0, 0) = range_sigma * range_sigma;
    r(1, 1) = bearing_sigma * bearing_sigma;

    const kalman::ExtendedKalmanFilter::StateTransitionFunction transition =
        [&a](const Eigen::VectorXd& x) -> Eigen::VectorXd { return a * x; };

    Eigen::VectorXd initial(4);
    initial << 80.0, 120.0, 1.5, -2.0;
    Eigen::Vector4d truth;
    truth << 100.0, 100.0, 1.5, -2.0;

    Eigen::Matrix4d initial_covariance = Eigen::Matrix4d::Zero();
    initial_covariance.diagonal() << 25.0, 25.0, 1.0, 1.0;

    kalman::ExtendedKalmanFilter filter(initial, initial_covariance);

    std::mt19937 generator(99);
    std::normal_distribution<double> range_noise(0.0, range_sigma);
    std::normal_distribution<double> bearing_noise(0.0, bearing_sigma);
    std::normal_distribution<double> process_noise(
        0.0, std::sqrt(acceleration_variance));

    for (int step = 0; step < 300; ++step) {
        truth = a * truth;
        truth(2) += process_noise(generator) * kDt;
        truth(3) += process_noise(generator) * kDt;

        const double range =
            std::hypot(truth(0), truth(1)) + range_noise(generator);
        const double bearing =
            std::atan2(truth(1), truth(0)) + bearing_noise(generator);
        Eigen::VectorXd measurement(2);
        measurement << range, bearing;

        CHECK_TRUE(filter.Predict(transition, q));
        CHECK_TRUE(filter.Update(BearingRangeMeasurement, measurement, r,
                                 BearingRangeJacobian));
        CHECK_TRUE(filter.covariance().allFinite());
        CHECK_TRUE(kalman::IsPositiveSemidefinite(filter.covariance()));
    }

    const Eigen::Vector2d position_error =
        filter.state().head<2>() - truth.head<2>();
    // The reported covariance must not be smaller than the actual error by
    // more than an order of magnitude (a "consistency" check): with 300 samples
    // this is a very loose bound, but it fails loudly on a filter that has
    // collapsed its covariance or diverged.
    const double mahalanobis =
        position_error.transpose() *
        filter.covariance().topLeftCorner<2, 2>().inverse() * position_error;
    CHECK_TRUE(mahalanobis < 100.0);
    CHECK_TRUE(position_error.norm() < 10.0);
}

TEST(ExtendedKalmanFilter, RejectsInvalidInputWithoutCorruptingState) {
    kalman::ExtendedKalmanFilter filter(Eigen::Vector2d::Zero(),
                                        Eigen::Matrix2d::Identity());
    const Eigen::VectorXd state_before = filter.state();
    const Eigen::MatrixXd covariance_before = filter.covariance();

    // Measurement of the wrong dimension.
    CHECK_FALSE(filter.Update(PendulumMeasurement, Eigen::VectorXd::Zero(3),
                              Eigen::MatrixXd::Identity(3, 3)));
    // Non-finite measurement.
    Eigen::VectorXd bad_measurement(1);
    bad_measurement << std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(filter.Update(PendulumMeasurement, bad_measurement,
                              MeasurementNoise()));
    // Non-positive-definite measurement noise.
    Eigen::MatrixXd bad_noise(1, 1);
    bad_noise << -1.0;
    CHECK_FALSE(filter.Update(PendulumMeasurement, Eigen::VectorXd::Zero(1),
                              bad_noise));
    // Process noise of the wrong size.
    CHECK_FALSE(filter.Predict(PendulumTransitionDynamic,
                               Eigen::MatrixXd::Identity(3, 3)));
    // Non-finite process noise.
    Eigen::Matrix2d bad_process_noise = Eigen::Matrix2d::Identity();
    bad_process_noise(0, 0) = std::numeric_limits<double>::infinity();
    CHECK_FALSE(filter.Predict(PendulumTransitionDynamic, bad_process_noise));
    // Negative-definite process noise.
    CHECK_FALSE(filter.Predict(PendulumTransitionDynamic,
                               -Eigen::Matrix2d::Identity()));

    CHECK_TRUE((filter.state() - state_before).norm() == 0.0);
    CHECK_TRUE((filter.covariance() - covariance_before).norm() == 0.0);

    // A model that produces garbage is refused too.
    const kalman::ExtendedKalmanFilter::StateTransitionFunction broken =
        [](const Eigen::VectorXd&) -> Eigen::VectorXd {
        Eigen::VectorXd bad(2);
        bad << std::numeric_limits<double>::quiet_NaN(), 0.0;
        return bad;
    };
    CHECK_FALSE(filter.Predict(broken, Eigen::Matrix2d::Zero()));
    CHECK_TRUE((filter.state() - state_before).norm() == 0.0);
}

TEST(ExtendedKalmanFilter, ResetValidatesCovariance) {
    kalman::ExtendedKalmanFilter filter;
    CHECK_TRUE(
        filter.Reset(Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity()));
    Eigen::Matrix2d indefinite;
    indefinite << 1.0, 0.0, 0.0, -1.0;
    CHECK_FALSE(filter.Reset(Eigen::Vector2d::Zero(), indefinite));
    // The failed reset must not have changed anything.
    CHECK_NEAR((filter.covariance() - Eigen::Matrix2d::Identity()).norm(), 0.0,
               0.0);
    // Wrong-sized covariance.
    CHECK_FALSE(
        filter.Reset(Eigen::Vector2d::Zero(), Eigen::MatrixXd::Identity(3, 3)));
}

TEST(ExtendedKalmanFilter, FrozenJacobianBecomesOverconfidentWhenNonlinear) {
    // The failure mode of "linearize once and ship it" is not a large raw
    // error, it is a covariance that stops describing the truth. The measurable
    // signature is the normalized estimation error squared,
    //
    //   NEES = e^T P^-1 e,  averaged over samples,
    //
    // which is ~2 for a consistent two-state estimator (chi-square with two
    // degrees of freedom) and much larger for a filter that claims more
    // accuracy than it has.
    //
    // The setup removes the damping so the pendulum keeps swinging at large
    // amplitude: with damping the 150 degree swing collapses to the linear
    // regime within a second, and then the frozen Jacobian is accidentally
    // right again, which is not what is being tested here.
    constexpr double kRelease = 150.0 * M_PI / 180.0;
    constexpr int kSteps = 400;
    constexpr int kSettlingSteps = 150;
    constexpr int kRuns = 40;

    const auto propagate = [](const Eigen::Vector2d& state) {
        constexpr int kPropagationSubSteps = 10;
        const double h = kSampleTime / kPropagationSubSteps;
        Eigen::Vector2d x = state;
        for (int i = 0; i < kPropagationSubSteps; ++i) {
            const auto undamped_dynamics = [](const Eigen::Vector2d& s) {
                return Eigen::Vector2d(
                    s(1), -kGravityOverLength * std::sin(s(0)));  // no damping
            };
            const Eigen::Vector2d k1 = undamped_dynamics(x);
            const Eigen::Vector2d k2 = undamped_dynamics(x + 0.5 * h * k1);
            const Eigen::Vector2d k3 = undamped_dynamics(x + 0.5 * h * k2);
            const Eigen::Vector2d k4 = undamped_dynamics(x + h * k3);
            x += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        }
        kalman::NormalizeAnglesInPlace(x.head<1>());
        return x;
    };
    const kalman::ExtendedKalmanFilter::StateTransitionFunction transition =
        [&propagate](const Eigen::VectorXd& x) -> Eigen::VectorXd {
        return propagate(x);
    };
    const Eigen::Matrix2d process_noise =
        (Eigen::Vector2d() << 1e-8, 1e-6).finished().asDiagonal();
    // Frozen Jacobian: exp(A_c Ts) with A_c taken at theta = 0.
    Eigen::Matrix2d frozen;
    {
        const double trace = 0.0;
        const double determinant = kGravityOverLength;
        const double omega = std::sqrt(determinant);
        frozen =
            std::cos(omega * kSampleTime) * Eigen::Matrix2d::Identity() +
            (std::sin(omega * kSampleTime) / omega) *
                (Eigen::Matrix2d() << 0.0, 1.0, -determinant, 0.0).finished();
        (void)trace;
    }

    double consistent_nees = 0.0;
    double frozen_nees = 0.0;
    for (int run = 0; run < kRuns; ++run) {
        std::mt19937 generator(13579u + static_cast<unsigned>(run));
        std::normal_distribution<double> measurement_noise(0.0, kAngleSigma);
        std::normal_distribution<double> process_noise_sample(0.0, 1.0);

        Eigen::MatrixXd initial_covariance = Eigen::MatrixXd::Zero(2, 2);
        initial_covariance.diagonal() << std::pow(2.0 * M_PI / 180.0, 2), 0.01;
        const Eigen::VectorXd initial_state =
            (Eigen::Vector2d() << kRelease, 0.0).finished();

        kalman::ExtendedKalmanFilter consistent(initial_state,
                                                initial_covariance);
        kalman::ExtendedKalmanFilter frozen_jacobian(initial_state,
                                                     initial_covariance);

        Eigen::Vector2d truth;
        truth << kRelease, 0.0;
        double run_nees_consistent = 0.0;
        double run_nees_frozen = 0.0;
        int counted = 0;

        for (int step = 0; step < kSteps; ++step) {
            truth = propagate(truth);
            truth(0) += 1e-4 * process_noise_sample(generator);
            truth(1) += 1e-3 * process_noise_sample(generator);

            Eigen::VectorXd measurement(1);
            measurement << truth(0) + measurement_noise(generator);

            CHECK_TRUE(consistent.Predict(transition, process_noise,
                                          PendulumTransitionJac));
            CHECK_TRUE(consistent.Update(PendulumMeasurement, measurement,
                                         MeasurementNoise(),
                                         PendulumMeasurementJacobian));
            CHECK_TRUE(frozen_jacobian.Predict(
                transition, process_noise,
                [&frozen](const Eigen::VectorXd&) -> Eigen::MatrixXd {
                    return frozen;
                }));
            CHECK_TRUE(frozen_jacobian.Update(PendulumMeasurement, measurement,
                                              MeasurementNoise(),
                                              PendulumMeasurementJacobian));

            if (step >= kSettlingSteps) {
                Eigen::Vector2d consistent_error;
                consistent_error << kalman::AngularDifference(
                    consistent.state()(0), truth(0)),
                    consistent.state()(1) - truth(1);
                Eigen::Vector2d frozen_error;
                frozen_error << kalman::AngularDifference(
                    frozen_jacobian.state()(0), truth(0)),
                    frozen_jacobian.state()(1) - truth(1);
                run_nees_consistent += consistent_error.transpose() *
                                       consistent.covariance().inverse() *
                                       consistent_error;
                run_nees_frozen += frozen_error.transpose() *
                                   frozen_jacobian.covariance().inverse() *
                                   frozen_error;
                ++counted;
            }
        }
        consistent_nees += run_nees_consistent / counted;
        frozen_nees += run_nees_frozen / counted;
    }
    consistent_nees /= kRuns;
    frozen_nees /= kRuns;

    // The honest EKF reports NEES close to the expected value of 2.
    CHECK_TRUE(consistent_nees > 0.7);
    CHECK_TRUE(consistent_nees < 3.5);
    // The frozen Jacobian is overconfident by a large factor (measured: ~7x).
    CHECK_TRUE(frozen_nees > 2.5 * consistent_nees);
}

TEST(ExtendedKalmanFilter, ReportsWhyAStepWasRefused) {
    // A refused step is safe; a silently refused step is a bug that hides for
    // weeks. Every rejection path must therefore say what went wrong, and the
    // nastiest one to hit by accident is easy to demonstrate: the transition
    // and measurement Jacobian callbacks have the same C++ type, so this
    // compiles and only fails at run time.
    kalman::ExtendedKalmanFilter filter(Eigen::Vector2d::Zero(),
                                        Eigen::Matrix2d::Identity());
    std::vector<kalman::FilterError> errors;
    filter.set_error_handler(
        [&errors](kalman::FilterError error) { errors.push_back(error); });

    const Eigen::MatrixXd state_before = filter.state();
    const Eigen::MatrixXd covariance_before = filter.covariance();

    // Measurement Jacobian (1x2) passed to Predict, which needs a 2x2.
    CHECK_FALSE(filter.Predict(PendulumTransitionDynamic, ProcessNoise(),
                               PendulumMeasurementJacobian));
    CHECK_EQ(errors.size(), std::size_t{1});
    CHECK_TRUE(errors.back() ==
               kalman::FilterError::kJacobianDimensionMismatch);
    CHECK_NEAR((filter.state() - state_before).norm(), 0.0, 0.0);
    CHECK_NEAR((filter.covariance() - covariance_before).norm(), 0.0, 0.0);

    // Non-finite measurement.
    Eigen::VectorXd not_a_number(1);
    not_a_number << std::numeric_limits<double>::quiet_NaN();
    CHECK_FALSE(
        filter.Update(PendulumMeasurement, not_a_number, MeasurementNoise()));
    CHECK_TRUE(errors.back() == kalman::FilterError::kNonFiniteInput);

    // Non-positive-semidefinite measurement noise.
    Eigen::MatrixXd negative_noise(1, 1);
    negative_noise << -1.0;
    CHECK_FALSE(filter.Update(PendulumMeasurement, Eigen::VectorXd::Zero(1),
                              negative_noise));
    CHECK_TRUE(errors.back() ==
               kalman::FilterError::kCovarianceNotPositiveSemidefinite);

    // A model that produces garbage.
    const kalman::ExtendedKalmanFilter::StateTransitionFunction broken =
        [](const Eigen::VectorXd&) -> Eigen::VectorXd {
        Eigen::VectorXd bad(2);
        bad << std::numeric_limits<double>::infinity(), 0.0;
        return bad;
    };
    CHECK_FALSE(filter.Predict(broken, ProcessNoise()));
    CHECK_TRUE(errors.back() == kalman::FilterError::kNonFiniteModelOutput);

    // A valid step afterwards must still work and must not report anything.
    const std::size_t error_count = errors.size();
    CHECK_TRUE(filter.Predict(PendulumTransitionDynamic, ProcessNoise(),
                              PendulumTransitionJac));
    CHECK_TRUE(filter.Update(PendulumMeasurement,
                             Eigen::VectorXd::Constant(1, 0.1),
                             MeasurementNoise(), PendulumMeasurementJacobian));
    CHECK_EQ(errors.size(), error_count);

    // The messages are meant to be readable in a log line.
    CHECK_TRUE(std::string(kalman::ToString(
                   kalman::FilterError::kJacobianDimensionMismatch)) != "");
    CHECK_TRUE(std::string(kalman::ToString(kalman::FilterError::kNone)) != "");
}

TEST(IteratedExtendedKalmanFilter, SingleIterationEqualsPlainEkf) {
    kalman::IteratedExtendedKalmanFilter iterated(Eigen::Vector2d::Zero(),
                                                  Eigen::Matrix2d::Identity());
    iterated.set_max_iterations(1);
    kalman::ExtendedKalmanFilter plain(Eigen::Vector2d::Zero(),
                                       Eigen::Matrix2d::Identity());

    const Eigen::Matrix2d process_noise = ProcessNoise();
    std::mt19937 generator(3);
    std::normal_distribution<double> noise(0.0, kAngleSigma);
    for (int step = 0; step < 25; ++step) {
        Eigen::VectorXd measurement(1);
        measurement << 0.3 + noise(generator);
        CHECK_TRUE(iterated.Predict(PendulumTransitionDynamic, process_noise));
        CHECK_TRUE(plain.Predict(PendulumTransitionDynamic, process_noise));
        CHECK_TRUE(iterated.UpdateIterated(PendulumMeasurement, measurement,
                                           MeasurementNoise(),
                                           PendulumMeasurementJacobian));
        CHECK_TRUE(plain.Update(PendulumMeasurement, measurement,
                                MeasurementNoise(),
                                PendulumMeasurementJacobian));
        CHECK_NEAR((iterated.state() - plain.state()).norm(), 0.0, 1e-12);
        CHECK_NEAR((iterated.covariance() - plain.covariance()).norm(), 0.0,
                   1e-12);
    }
    CHECK_EQ(iterated.last_iterations(), 1);
}

TEST(IteratedExtendedKalmanFilter,
     ConvergesAndImprovesOnStronglyNonlinearUpdate) {
    // A large angle measurement far from the current estimate: the iterated
    // update must converge (Gauss-Newton) and still return a valid covariance.
    kalman::IteratedExtendedKalmanFilter filter(
        Eigen::Vector2d::Zero(), Eigen::Matrix2d::Identity() * 0.01);
    filter.set_max_iterations(10);

    Eigen::VectorXd measurement(1);
    measurement << 1.4;  // 80 degrees, far from 0
    CHECK_TRUE(filter.Predict(PendulumTransitionDynamic, ProcessNoise()));
    CHECK_TRUE(filter.UpdateIterated(PendulumMeasurement, measurement,
                                     MeasurementNoise(),
                                     PendulumMeasurementJacobian));
    CHECK_TRUE(filter.last_iterations() >= 2);
    CHECK_TRUE(filter.last_iterations() <= 10);
    CHECK_TRUE(filter.state().allFinite());
    CHECK_TRUE(kalman::IsPositiveSemidefinite(filter.covariance()));
    // The estimate must move towards the measurement, not past it.
    CHECK_TRUE(filter.state()(0) > 0.0);
    CHECK_TRUE(filter.state()(0) <= 1.4 + 1e-9);
}
