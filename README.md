# Kalman filter library (C++17)

[![ci](https://github.com/xingxingRealzyx/kalman-filter/actions/workflows/ci.yml/badge.svg)](https://github.com/xingxingRealzyx/kalman-filter/actions/workflows/ci.yml)

A small, self-contained state estimation library: linear Kalman filter, extended
Kalman filter (with the iterated variant), and unscented Kalman filter. Eigen for
linear algebra, no other dependencies, C++17, Google style with four-space
indentation.

The design goal is not "the most features" but "does not lie to you": every
number the filter reports is defensible, every input is validated, and every
refused step says why.

```
include/kalman/                <- the library's public API
  linear_kalman_filter.h       LinearKalmanFilter<N, M, NU>   (header-only)
  extended_kalman_filter.h     ExtendedKalmanFilter, IteratedExtendedKalmanFilter
  unscented_kalman_filter.h    UnscentedKalmanFilter
  numerics.h                   symmetry / PSD / angle / gain-solve helpers
  attitude/                    <- optional module, header-only (kalman::attitude)
    quaternion.h               quaternion / rotation helpers, verified Jacobians
    imu_attitude_model.h       gyro + accelerometer attitude model, static calibration
src/                           implementations of the three filters
examples/                      linear_tracking_example, pendulum_ekf_example,
                               imu_attitude_example
tests/                         test framework + 50 unit tests
third_party/eigen/             vendored Eigen 3.4.0 (used when no system Eigen)
```

Two targets, so that a model is opt-in rather than part of the library:

| target | contents | who needs it |
|---|---|---|
| `kalman::kalman` | the three filters and `numerics.h` | anyone estimating anything |
| `kalman::attitude` | `include/kalman/attitude/` (quaternion math + the IMU model) | attitude estimation |

The IMU model is application code, not filter code: if you are tracking a target
or estimating a state of charge, you should not have to compile it or read around
it in the library's headers. It is header-only, so opting in costs one include:

```cpp
#include "kalman/attitude/imu_attitude_model.h"   // and link kalman::attitude
```

## Build and run

Requires CMake 3.21+ (for `CMakePresets.json`) and a C++17 compiler. Eigen is
picked up from the system when available (`brew install eigen`), otherwise the
vendored copy under `third_party/eigen` is used, so nothing has to be installed.

```bash
cmake --preset release            # or: debug, ci, sanitizers
cmake --build --preset release -j
ctest --preset release            # 50 tests

./build/release/linear_tracking_example
./build/release/pendulum_ekf_example
./build/release/imu_attitude_example
```

| preset | what it is for |
|---|---|
| `release` | optimized build with the strict warning set |
| `debug` | `-O0 -g`; Eigen's internal assertions are active |
| `ci` | `release` plus `-Werror` (what CI builds) |
| `sanitizers` | `debug` plus AddressSanitizer and UndefinedBehaviorSanitizer |

Without presets, the equivalent of `release` is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

The library builds warning-free with `-Wall -Wextra -Wpedantic -Wshadow
-Wconversion -Wsign-conversion -Wold-style-cast -Wdouble-promotion -Wformat=2`;
`-DKALMAN_WERROR=ON` turns warnings into errors.

## Quick start

Linear filter, fixed size, on the stack (embedded friendly):

```cpp
kalman::LinearKalmanFilter<4, 2> filter(x0, P0);   // [px py vx vy], measure position
filter.Predict(A, Q);                              // x = A x,  P = A P A' + Q
filter.Update(C, measurement, R);                  // Kalman update, returns bool
```

Extended filter, dynamic size, model as lambdas:

```cpp
kalman::ExtendedKalmanFilter filter(x0, P0);
filter.Predict(transition, Q);                     // Jacobian by finite differences
filter.Update(measurement_function, y, R);         // ditto for the measurement
```

The Jacobian arguments are optional. When a model is cheap to evaluate, letting
the library take a central difference of it is usually the better trade: less
code, no sign errors, and the cost is one model evaluation per state component.
Pass a Jacobian when you have an analytic one you trust.

## What each filter does

### Linear Kalman filter

For the model

$$
x_k = A x_{k-1} + B u_{k-1} + w, \qquad y_k = C x_k + v, \qquad w \sim N(0, Q),\ v \sim N(0, R)
$$

the recursion is the textbook one. Two details are worth calling out because
they are where implementations usually cut corners:

* The gain is computed by *solving* with the innovation covariance, never by
  forming an inverse. `SolveKalmanGain` first attempts an exact Cholesky solve
  and checks the residual of the result; only an exactly singular system falls
  back to a regularized retry, and that path is reported as a refused step.
  Adding jitter to $S$ "just in case" is a silent bias on every covariance the
  filter reports, so it is not done here.
* The covariance update defaults to the Joseph form,
  $P = (I - KC)P^- (I - KC)^\top + KRK^\top$, which is symmetric by construction
  and stays positive semidefinite for any gain. `CovarianceUpdate::kStandard`
  selects the cheaper textbook form when $R$ is well conditioned.

### Extended Kalman filter

Same recursion on the first-order expansion of the nonlinear model:

$$
x_k^- = f(x_{k-1}),\quad F = \left.\frac{\partial f}{\partial x}\right|_{x_{k-1}},\quad P_k^- = F P_{k-1} F^\top + Q
$$

$$
S = H P^- H^\top + R,\quad K = P^- H^\top S^{-1},\quad x_k = x_k^- + K\left(y_k - h(x_k^-)\right)
$$

Angles are the caller's responsibility: call `kalman::NormalizeAnglesInPlace()`
inside the model functions, including inside the transition, so the finite
difference lands on the correct side of the wrap. `kalman::AngularDifference()`
does the same for error statistics.

`IteratedExtendedKalmanFilter::UpdateIterated()` re-linearizes the measurement
around each Gauss-Newton iterate instead of only at the predicted state. It is
the right tool when the *measurement* is strongly nonlinear (range, bearing,
polar-to-Cartesian); on the pendulum below it changes nothing, which is itself
informative.

### Unscented Kalman filter

Propagates $2n+1$ sigma points through the true nonlinear functions and recovers
mean and covariance from the transformed points, with the scaled transform
$\lambda = \alpha^2(n + \kappa) - n$, $\kappa = 3 - n$, $\alpha = 10^{-3}$,
$\beta = 2$. No Jacobians, second-order accurate (third for symmetric
distributions), and the better default when the transition nonlinearity is
strong. `PredictLinear` / `CorrectLinear` mirror the matrix API for linear
models, which is also how the tests pin the UKF against the linear filter.

## Numerical behaviour under stress

The library is defensive on purpose, because a Kalman filter rarely fails
loudly. Every step validates its inputs, and a step that cannot be completed
leaves the filter exactly as it was:

* non-finite or wrongly shaped measurements, $Q$, $R$, model outputs, Jacobians;
* $Q$ and $R$ that are not positive semidefinite;
* innovation covariances that are singular in the direction of the innovation;
* a model function that returns the wrong number of states.

`ExtendedKalmanFilter::set_error_handler()` turns those refusals into
diagnostics:

```cpp
filter.set_error_handler([](kalman::FilterError error) {
  std::cerr << "EKF refused a step: " << kalman::ToString(error) << "\n";
});
```

This exists because of a sharp edge in the type system: `StateTransitionJacobian`
and `MeasurementJacobian` are both `std::function<MatrixXd(const VectorXd&)>`,
so passing a $1\times n$ measurement Jacobian where an $n\times n$ transition
Jacobian belongs **compiles without a warning** and produces a covariance that
is simply wrong. `Predict()` checks the shape at run time and reports
`FilterError::kJacobianDimensionMismatch`. (This is not hypothetical: it
happened while writing the tests for this library, and the resulting failure was
a filter that refused every step with no explanation.)

## Measured results

Both examples run deterministic Monte-Carlo sweeps with fixed seeds, so these
numbers are reproducible.

### Linear tracking (10 Hz position sensor, 2 m sigma)

A constant-velocity target tracked with a white-noise-acceleration model
converges to 0.8 m position error and 0.10 m/s velocity error, against a
reported 1-sigma of 0.55 m and 0.27 m/s: the estimate is consistent with the
covariance that the filter claims. The covariance also settles to the fixed
point of the discrete algebraic Riccati equation (verified directly in the test
suite, along with a Monte-Carlo check that the reported covariance matches the
empirical error covariance across 200 runs).

### Pendulum with an angle-only sensor

Nonlinear model, only the angle is measured, the rate must be inferred:
$\ddot\theta = -(g/l)\sin\theta - b\dot\theta$, $y = \theta + \nu$. Thirty runs of
300 steps, 100 settling steps discarded, process noise from a continuous
white-noise acceleration with $\sigma_a = 1\ \mathrm{rad/s^2}$.

Released from 30 degrees (inside the linear regime):

| estimator | angle RMSE | rate RMSE | NEES | reported 1-sigma |
|---|---|---|---|---|
| EKF, finite-difference Jacobian | 4.94 deg | 15.42 deg/s | 1.80 | 5.35 deg |
| EKF, `exp(A_c*Ts)` Jacobian | 4.94 deg | 15.42 deg/s | 1.80 | 5.34 deg |
| IEKF, iterated update | 4.94 deg | 15.42 deg/s | 1.80 | 5.35 deg |
| UKF, no Jacobian | 4.94 deg | 15.46 deg/s | 1.80 | 5.35 deg |
| EKF, Jacobian frozen at 0 | 0.36 deg | 4.19 deg/s | 2.01 | 0.36 deg |

Released from 150 degrees (far outside it):

| estimator | angle RMSE | rate RMSE | NEES | reported 1-sigma |
|---|---|---|---|---|
| EKF, finite-difference Jacobian | 5.26 deg | 15.46 deg/s | 1.62 | 6.56 deg |
| UKF, no Jacobian | 5.25 deg | 15.40 deg/s | 1.55 | 6.54 deg |
| EKF, Jacobian frozen at 0 | 0.52 deg | 7.83 deg/s | **4.78** | 0.36 deg |

The interesting row is the last one, and the interesting column is NEES
(normalized estimation error squared, $e^\top P^{-1} e$, where ~2 is the
consistent value for two states). The frozen-Jacobian filter looks *better* on
raw RMSE while being overconfident by a factor of three: it reports 0.36 degrees
of uncertainty whether the pendulum is at 30 degrees or 150, because its
covariance is propagated with $A = \exp(A_c(0)T_s)$, whose eigenvalues sit on
the unit circle. It wins on RMSE partly by ignoring its measurements. On a real
system, where the model is never exactly the truth, that same overconfidence is
how a filter walks away from reality without any warning. Two conclusions:

* Compare filters by consistency (NEES, or the innovation whiteness test), not
  by RMSE alone. RMSE rewards a filter for being timid.
* The frozen Jacobian is the failure mode worth understanding; at 30 degrees it
  is harmless, at 150 it is the whole problem.

## The IMU example: what a gyro and accelerometer can and cannot see

`imu_attitude_example` estimates ZYX Euler angles from a gyroscope (prediction) and
an accelerometer (correction), with an optional magnetometer. The model lives in
`include/kalman/attitude/imu_attitude_model.h` so it can be reused; the state is the attitude
quaternion alone, and the sensor biases are measured by a static calibration rather
than estimated online.

Measured over the 100 Hz, 20 s scenario (static 0-5 s, rotating 5-15 s, static
15-20 s), all with fixed seeds:

| estimator | roll RMS | pitch RMS | yaw RMS | reported 1-sigma |
|---|---|---|---|---|
| gyro + accelerometer | 1.70 deg | 1.69 deg | 16.68 deg | 0.048 deg |
| + magnetometer | 1.07 deg | 0.50 deg | 0.18 deg | - |

Three things this makes concrete:

* **Gravity fixes two directions, not three.** A rotation about the vertical leaves
  the accelerometer reading unchanged, which the test suite asserts by showing that
  a 5 degree yaw change moves the measurement by less than $10^{-12}$. Heading then
  accumulates whatever the gyro bias got wrong, and here that is 17 degrees over
  20 seconds: physics, not a tuning failure.
* **A magnetometer is the missing reference**, and it takes heading from 16.7
  degrees of RMS error to 0.18.
* **The reported covariance is a noise floor, not an error bound.** It settles at
  0.048 degrees because averaging 100 Hz of unbiased noise really would give that;
  the actual error is a degree because linear acceleration is a *bias* over the
  timescale of a manoeuvre, and averaging does not remove a bias.

Online bias estimation was tried first and removed. An accelerometer bias is nearly
degenerate with attitude (0.6 degrees of tilt looks like 0.1 m/s^2 of bias), and
with the bias in the state the two feed each other until the filter walks away. The
static calibration in the attitude module is both cheaper and more robust.

## Four bugs found while building this (and fixed)

These are recorded because each one is a trap that is easy to fall into and hard
to see.

1. **`SymmetrizeInPlace` aliased its own destination.** The obvious
   `A = (A + A.transpose()) * 0.5` evaluates coefficient by coefficient, so the
   transpose reads elements of `A` that have already been overwritten. The
   matrix comes out *not* symmetric (off by ~1.7 in a test case) and every
   covariance downstream is quietly wrong. Now the transpose goes into a
   temporary first.
2. **Regularizing the innovation covariance biased it.** Adding jitter to $S$
   before the solve perturbs the gain and hence the reported covariance, and it
   also hid a genuine correctness question: a singular $S$ is either
   "no information in this direction" (gain legitimately zero) or "undefined
   0/0" (must refuse). The solver now decides by checking the residual
   $SK^\top = X^\top$ and only falls back to a regularized retry for the second
   case, which it reports as a refused step.
3. **`std::remainder` maps a half turn to $-\pi$, not $+\pi$** (it rounds the
   quotient to even). A filter reporting $-180°$ where the caller expects
   $+180°$ looks like a full-turn jump to every consumer downstream.
   `NormalizeAnglesInPlace` now normalizes to $(-\pi, \pi]$.
4. **The "analytic Jacobian" shortcut is not the discrete Jacobian.** For
   $\exp(A_c T_s)$ on the pendulum the error is ~$7\times10^{-4}$ relative,
   because $A_c(x(t))$ varies along the step; it is a frozen-Jacobian
   approximation, not an analytic answer. Finite differences of the same model
   function recover the true Jacobian of the discrete map to ~$10^{-9}$, which
   is verified in the test suite against an integrated variational equation.
   The test asserts that the shortcut is measurably wrong, so nobody
   "optimizes" it back in.

## Tests

50 tests, all deterministic (fixed seeds), run in well under a second:

| area | what is pinned down |
|---|---|
| linear filter | matches an independent textbook recursion step by step; steady-state covariance satisfies the Riccati equation and the recursion's two-step fixed point; reported covariance matches a 200-run Monte-Carlo error covariance; Joseph and standard forms agree; 20k steps stay symmetric and PSD with $R=10^{-6}$; invalid inputs are refused without touching the state |
| extended filter | on a linear model it reproduces the linear filter to machine precision (with and without Jacobians); finite-difference Jacobian equals the integrated variational Jacobian; pendulum tracking; bearing-range consistency; the frozen-Jacobian variant is shown to be overconfident via NEES; every refusal path reports its reason through the error handler |
| iterated filter | one iteration equals the plain EKF exactly; the Gauss-Newton loop converges on a strongly nonlinear update and never overshoots the measurement |
| unscented filter | matches the linear filter on linear models; the transform reproduces the input mean and covariance under identity and rotation; pendulum tracking without Jacobians; wrapped-angle averaging; bearing-range consistency |
| numerics | symmetry, PSD detection (including the round-off tolerance), singular regularization, angle wrapping, angle differences across the $\pm\pi$ boundary |
| quaternion | Hamilton product against Eigen, left/right multiplication matrices, rotation matrix orthonormality, the radial identity $J q = 0$ for both Jacobians, both Jacobians against central differences, ZYX Euler round trips across all quadrants |
| attitude (IMU) | quaternion propagation matches an exact axis-angle rotation, tangent-space property of the transition Jacobian, accelerometer/magnetometer Jacobians against central differences, instantaneous observability rank (gravity fixes two directions, magnetometer adds the third), static calibration accuracy, end-to-end roll/pitch tracking plus the yaw-drift comparison |

## Continuous integration

`.github/workflows/ci.yml` runs on every push to `main` and on every pull
request:

* **Ubuntu and macOS**, each in `ci` (`-Werror`) and `debug` configurations:
  configure, build, run all 50 tests through CTest, then run all three examples.
  Running the examples matters - they are the documented usage, so a broken
  example is a broken README.
* **AddressSanitizer + UndefinedBehaviorSanitizer** on Ubuntu (the `sanitizers`
  preset, with LeakSanitizer enabled). This is the job that earns its keep: it is
  what catches an out-of-bounds write or an uninitialized read inside a
  covariance update, which a plain build would happily run past.

Everything is driven by the same `CMakePresets.json` used locally, so CI cannot
drift away from what a developer runs by hand.

Three things CI found that no local build did, which is the whole argument for
having it:

* **A vendored dependency that was not actually in the repository.** Eigen's own
  `.gitignore` contains the bare rule `core`, which excludes `Eigen/Core` and
  `Eigen/src/Core/` on a case-sensitive checkout. The machine that vendored the
  code still had the files on disk, so every local build passed; a fresh Linux
  checkout did not have them at all. Fixed by dropping the vendored `.gitignore`;
  `VENDORED.md` explains how to avoid reintroducing it.
* **A shadowed constructor parameter** (`FailureException(const std::string&
  message) : message(message)`) that Clang accepts and GCC rejects under
  `-Wshadow`.
* **Two assertions that were too tight for a second compiler.** The same
  deterministic run differs by ~0.5 degrees between AppleClang and GCC 13, which
  was enough to flip a 2 degree threshold. They are now expressed as multiples of
  the sensor noise floor plus comparisons between axes, which is both more
  meaningful and stable across compilers.

Because of that last one, the Linux matrix is also reproduced locally with
containers while iterating, instead of pushing and waiting:

```bash
docker run --rm --platform linux/amd64 -v "$PWD":/src -w /src gcc:13 \
  bash -c "apt-get update -qq && apt-get install -y -qq cmake && \
           cmake -S . -B /tmp/b -DCMAKE_BUILD_TYPE=Release -DKALMAN_WERROR=ON && \
           cmake --build /tmp/b -j && cd /tmp/b && ctest"
```

## Limitations

* Double precision only by default; the fixed-size filter is templated on the
  scalar type, but `numerics.h` helpers assume `double`.
* No square-root or UD-factorized covariance propagation. The Joseph form plus
  symmetrization is enough for the conditioning this library was tested on; a
  square-root filter is the next step if you need very long runs with extreme
  $R/Q$ ratios.
* The UKF uses $\alpha = 10^{-3}$ by default, which is accurate but not
  bias-optimal for large state dimensions; set `set_alpha()` if you tune it.
* Angles are not tracked automatically: the model must call
  `NormalizeAnglesInPlace`, because only the caller knows which components wrap.
* No adaptive noise estimation (IAE/MMA), no multi-hypothesis or particle
  filters.

## A note on rotation Jacobians

`include/kalman/attitude/quaternion.h` derives its Jacobians by central differences
rather than in closed form, and that is a deliberate retreat. A closed form for $\partial (R(q)v)/\partial q$
was written by hand four times, and every version was wrong in a different way (a
sign on the scalar column, two missing terms, a missing factor of two in the
$SO(3)$ left Jacobian, and a radial projection applied on the wrong side). Each
wrong version passed the "looks like a rotation Jacobian" smell test and silently
corrupted the attitude covariance; one of them cost an afternoon of closed-loop
divergence before the homogeneity identity below found it.

Two invariants pin a correct implementation down, and the finite-difference version
satisfies both to machine precision:

* $J q = 0$: the rotation does not depend on the quaternion norm, so the derivative
  must annihilate the radial direction. This single line catches most sign errors.
* $J$ must equal a central difference of the model function along the *renormalized*
  path, which is the path a filter actually takes.

The same lesson applies to the propagation Jacobian: the exact $SO(3)$ left Jacobian
contributes $T_s[\tfrac12\cos(a/2)uu^\top + (\sin(a/2)/a)(I - uu^\top)]$, and the
perpendicular coefficient $\sin(a/2)/a$ is easy to get wrong by a factor of two,
which understates the gyro-bias-to-attitude coupling by that factor.

## Formatting

`.clang-format` in this directory encodes the project style (Google base,
`IndentWidth: 4`, 80 columns):

```bash
clang-format -i --style=file include/kalman/*.h src/*.cpp tests/*.cpp examples/*.cpp
```
