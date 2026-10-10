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

#ifndef KALMAN_TESTS_TESTING_H_
#define KALMAN_TESTS_TESTING_H_

// Minimal dependency-free test framework for this library.
//
//   TEST(LinearKalmanFilter, MatchesTextbookRecursion) {
//     CHECK_NEAR(actual, expected, 1e-12);
//     CHECK_TRUE(filter.Update(c, y, r));
//   }
//
// A failing CHECK records a message and returns from the test; the runner
// prints a summary and exits non-zero when anything failed.

#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace kalman_test {

struct TestCase {
    std::string name;
    std::function<void()> function;
};

// Non-template (functions in testing.cpp), so tests can be split over several
// translation units without ODR or static-init-order surprises.
void RegisterTest(const std::string& name, std::function<void()> function);
void ReportFailure(const std::string& message);
int RunAllTests(const std::string& filter);

// Thrown by a failing CHECK to abort the current test.
struct FailureException {
    explicit FailureException(const std::string& text) : message(text) {}
    std::string message;
};

struct TestRegistrar {
    TestRegistrar(const std::string& name, std::function<void()> function) {
        RegisterTest(name, function);
    }
};

// Message builder: use like `return Fail() << "saw " << value;` or
// `CHECK_TRUE(x) << "context"`. Kept simple on purpose.
class Message {
 public:
    template <typename T>
    Message& operator<<(const T& value) {
        stream_ << value;
        return *this;
    }
    std::string str() const { return stream_.str(); }

 private:
    std::ostringstream stream_;
};

}  // namespace kalman_test

#define KALMAN_TEST_CONCAT_INNER(a, b) a##b
#define KALMAN_TEST_CONCAT(a, b) KALMAN_TEST_CONCAT_INNER(a, b)

#define TEST(suite, name)                                                 \
    static void KALMAN_TEST_CONCAT(suite, KALMAN_TEST_CONCAT(_, name))(); \
    static ::kalman_test::TestRegistrar KALMAN_TEST_CONCAT(               \
        registrar_, KALMAN_TEST_CONCAT(suite, name))(                     \
        #suite "." #name,                                                 \
        &KALMAN_TEST_CONCAT(suite, KALMAN_TEST_CONCAT(_, name)));         \
    static void KALMAN_TEST_CONCAT(suite, KALMAN_TEST_CONCAT(_, name))()

#define KALMAN_TEST_FAIL_AT(file, line)                        \
    ::kalman_test::ReportFailure(std::string(file) + ":" +     \
                                 std::to_string(line) + ": " + \
                                 std::string("check failed"))

#define CHECK_TRUE(condition)                                                 \
    do {                                                                      \
        if (!(condition)) {                                                   \
            ::kalman_test::Message kalman_test_message;                       \
            kalman_test_message << "CHECK_TRUE(" #condition ") failed";       \
            KALMAN_TEST_FAIL_AT(__FILE__, __LINE__);                          \
            throw ::kalman_test::FailureException(kalman_test_message.str()); \
        }                                                                     \
    } while (false)

#define CHECK_FALSE(condition) CHECK_TRUE(!(condition))

#define CHECK_EQ(actual, expected)                                            \
    do {                                                                      \
        const auto& kalman_test_actual = (actual);                            \
        const auto& kalman_test_expected = (expected);                        \
        if (!(kalman_test_actual == kalman_test_expected)) {                  \
            ::kalman_test::Message kalman_test_message;                       \
            kalman_test_message                                               \
                << "CHECK_EQ(" #actual ", " #expected ") failed: got "        \
                << kalman_test_actual << ", want " << kalman_test_expected;   \
            KALMAN_TEST_FAIL_AT(__FILE__, __LINE__);                          \
            throw ::kalman_test::FailureException(kalman_test_message.str()); \
        }                                                                     \
    } while (false)

#define CHECK_NEAR(actual, expected, tolerance)                               \
    do {                                                                      \
        const double kalman_test_actual = static_cast<double>(actual);        \
        const double kalman_test_expected = static_cast<double>(expected);    \
        const double kalman_test_tolerance = (tolerance);                     \
        if (!(std::abs(kalman_test_actual - kalman_test_expected) <=          \
              kalman_test_tolerance)) {                                       \
            ::kalman_test::Message kalman_test_message;                       \
            kalman_test_message                                               \
                << "CHECK_NEAR(" #actual ", " #expected ") failed: got "      \
                << kalman_test_actual << ", want " << kalman_test_expected    \
                << ", tolerance " << kalman_test_tolerance;                   \
            KALMAN_TEST_FAIL_AT(__FILE__, __LINE__);                          \
            throw ::kalman_test::FailureException(kalman_test_message.str()); \
        }                                                                     \
    } while (false)

#endif  // KALMAN_TESTS_TESTING_H_
