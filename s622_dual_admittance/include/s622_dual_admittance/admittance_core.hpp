#pragma once

#include <array>
#include <cstddef>

namespace s622_dual_admittance
{

using Vector6 = std::array<double, 6>;

/**
 * @brief 导纳控制器参数
 *
 * 详细说明：
 * 质量、阻尼、刚度、死区、最大速度和最大位移都是6维向量，分别对应于6个自由度。
 */
struct AdmittanceParameters
{
  Vector6 mass;
  Vector6 damping;
  Vector6 stiffness;
  Vector6 deadband;             // N, N, N, N*m, N*m, N*m
  Vector6 max_velocity;         // m/s, m/s, m/s, rad/s, rad/s, rad/s
  Vector6 max_displacement;     // m, m, m, rad, rad, rad
  std::array<bool, 6> enabled_axes;
  bool limits_enabled{true};
};

class AdmittanceCore
{
public:
  explicit AdmittanceCore(const AdmittanceParameters & parameters);

  Vector6 update(
    const Vector6 & external_wrench,
    const Vector6 & desired_wrench,
    double dt);

  void reset();

  const Vector6 & displacement() const;
  const Vector6 & velocity() const;

private:
  static double apply_deadband(double value, double deadband);

  AdmittanceParameters parameters_;
  Vector6 displacement_{};
  Vector6 velocity_{};
};

}  // namespace s622_dual_admittance
