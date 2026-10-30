// Copyright 2026 dsl
// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <iostream>
#include "../src/joint_position_limiter.hpp"

int main()
{
  s622_dual_admittance::JointPositionLimiter limiter;
  const std::vector<double> measured{0.0, 1.0};
  const std::vector<double> target{0.02, 0.98}, rates{1.0, 0.5};
  std::vector<double> previous = measured;
  for (int tick = 0; tick < 20; ++tick) {
    // A rigid surface keeps measured positions fixed. The commanded preload
    // must still reach the target, at a bounded speed, without overshoot.
    const auto command = limiter.update(target, measured, rates, 0.004);
    for (std::size_t i = 0; i < command.size(); ++i) {
      if (std::abs(command[i] - previous[i]) > rates[i] * 0.004 + 1.e-12) {
        std::cerr << "commanded joint speed exceeded\n";
        return 1;
      }
    }
    previous = command;
  }
  if (std::abs(previous[0] - target[0]) > 1.e-12 ||
    std::abs(previous[1] - target[1]) > 1.e-12)
  {
    std::cerr << "static preload was incorrectly limited by measured position\n";
    return 1;
  }
  limiter.reset();
  const auto restarted = limiter.update(target, {0.01, 0.99}, rates, 0.004);
  if (std::abs(restarted[0] - 0.014) > 1.e-12 ||
    std::abs(restarted[1] - 0.988) > 1.e-12)
  {
    std::cerr << "restart did not seed from current feedback\n";
    return 1;
  }
  return 0;
}
