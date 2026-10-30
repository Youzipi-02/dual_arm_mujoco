// Copyright 2026 dsl
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <algorithm>
#include <vector>

namespace s622_dual_admittance
{

// Limit commanded joint speed without limiting the static position preload.
// Reset after disabling or losing feedback; the next command starts at feedback.
class JointPositionLimiter
{
public:
  void reset() {previous_.clear();}

  std::vector<double> update(
    const std::vector<double> & target, const std::vector<double> & measured,
    const std::vector<double> & max_velocity, double dt)
  {
    if (previous_.empty()) {
      previous_ = measured;
    }
    for (std::size_t i = 0; i < target.size(); ++i) {
      const double step = max_velocity.at(i) * dt;
      previous_.at(i) += std::clamp(target[i] - previous_.at(i), -step, step);
    }
    return previous_;
  }

private:
  std::vector<double> previous_;
};
}  // namespace s622_dual_admittance
