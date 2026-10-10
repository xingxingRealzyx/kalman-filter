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

#include "tests/testing.h"

#include <iostream>

namespace kalman_test {
namespace {

std::map<std::string, TestCase>& Registry() {
    static std::map<std::string, TestCase> registry;
    return registry;
}

struct RunState {
    int passed = 0;
    int failed = 0;
    std::vector<std::string> failures;
};

RunState& State() {
    static RunState state;
    return state;
}

}  // namespace

void RegisterTest(const std::string& name, std::function<void()> function) {
    Registry()[name] = TestCase{name, std::move(function)};
}

void ReportFailure(const std::string& message) {
    State().failures.push_back(message);
}

int RunAllTests(const std::string& filter) {
    if (Registry().empty()) {
        std::cerr << "No tests were registered.\n";
        return 1;
    }
    std::cout << "Running " << Registry().size() << " test(s)\n";
    for (auto& entry : Registry()) {
        const std::string& name = entry.first;
        if (!filter.empty() && name.find(filter) == std::string::npos) {
            continue;
        }
        const std::size_t failures_before = State().failures.size();
        std::cout << "[ RUN      ] " << name << "\n";
        try {
            entry.second.function();
        } catch (const FailureException& failure) {
            // The message was already recorded by the failing CHECK.
        } catch (const std::exception& error) {
            State().failures.push_back(
                name + ": unexpected exception: " + error.what());
        } catch (...) {
            State().failures.push_back(name +
                                       ": unexpected non-standard exception");
        }
        if (State().failures.size() == failures_before) {
            ++State().passed;
            std::cout << "[       OK ] " << name << "\n";
        } else {
            ++State().failed;
            std::cout << "[  FAILED  ] " << name << "\n";
            for (std::size_t i = failures_before; i < State().failures.size();
                 ++i) {
                std::cout << "             " << State().failures[i] << "\n";
            }
        }
    }
    std::cout << "\n"
              << State().passed << " passed, " << State().failed << " failed\n";
    return State().failed == 0 ? 0 : 1;
}

}  // namespace kalman_test

int main(int argc, char** argv) {
    const std::string filter = argc > 1 ? argv[1] : std::string();
    return kalman_test::RunAllTests(filter);
}
