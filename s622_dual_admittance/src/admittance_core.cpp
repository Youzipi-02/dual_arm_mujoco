#include "s622_dual_admittance/admittance_core.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace s622_dual_admittance
{


/**
 * @brief AdmittanceCore类的构造函数，检查导纳控制器参数的有效性
 *
 * @param parameters 导纳控制器参数
 */
AdmittanceCore::AdmittanceCore(
  const AdmittanceParameters & parameters)
: parameters_(parameters)
{
  for (std::size_t i = 0; i < 6; ++i) {
    if (!std::isfinite(parameters_.mass[i]) ||
      parameters_.mass[i] <= 0.0)
    {
      throw std::invalid_argument("mass must be finite and positive");
    }

    if (!std::isfinite(parameters_.damping[i]) ||
      parameters_.damping[i] < 0.0)
    {
      throw std::invalid_argument(
              "damping must be finite and non-negative");
    }

    if (!std::isfinite(parameters_.stiffness[i]) ||
      parameters_.stiffness[i] < 0.0)
    {
      throw std::invalid_argument(
              "stiffness must be finite and non-negative");
    }

    if (!std::isfinite(parameters_.deadband[i]) ||
      parameters_.deadband[i] < 0.0)
    {
      throw std::invalid_argument(
              "deadband must be finite and non-negative");
    }

    if (parameters_.limits_enabled) {
      if (!std::isfinite(parameters_.max_velocity[i]) ||
        parameters_.max_velocity[i] <= 0.0)
      {
        throw std::invalid_argument(
                "max_velocity must be finite and positive");
      }

      if (!std::isfinite(parameters_.max_displacement[i]) ||
        parameters_.max_displacement[i] <= 0.0)
      {
        throw std::invalid_argument(
                "max_displacement must be finite and positive");
      }
    }
  }
}

/**
 * @brief 根据导纳控制方程计算速度和位移的变化量
 *
 * @param external_wrench 抓取坐标系中的外力和外力矩
 * @param desired_wrench 期望力和期望力矩
 * @param dt 控制周期
 * @return displacement_ 位移变化量
 */
Vector6 AdmittanceCore::update(
  const Vector6 & external_wrench,
  const Vector6 & desired_wrench,
  const double dt)
{
  // 检查控制周期的有效性
  if (!std::isfinite(dt) || dt <= 0.0 || dt > 0.02) {
    throw std::invalid_argument("invalid control period");
  }

  // 对每个自由度进行导纳控制计算
  for (std::size_t i = 0; i < 6; ++i) {
    if (!parameters_.enabled_axes[i]) {
      displacement_[i] = 0.0;
      velocity_[i] = 0.0;
      continue;
    }

    // 检查外力和期望力的有效性
    if (!std::isfinite(external_wrench[i]) ||
      !std::isfinite(desired_wrench[i]))
    {
      reset();
      throw std::invalid_argument("wrench contains NaN or Inf");
    }

    double wrench_error = external_wrench[i] - desired_wrench[i];   // 计算力误差

    wrench_error = apply_deadband(wrench_error, parameters_.deadband[i]); // 应用死区处理

    // 后向欧拉同时求解阻尼和刚度，避免高阻尼轴在控制周期内交替反向。
    // m(v_next-v)/dt = F - d*v_next - k*x_next,
    // x_next = x + dt*v_next.
    const double mass = parameters_.mass[i];
    const double damping = parameters_.damping[i];
    const double stiffness = parameters_.stiffness[i];
    velocity_[i] =
      (mass * velocity_[i] + dt * (wrench_error - stiffness * displacement_[i])) /
      (mass + dt * damping + dt * dt * stiffness);

    if (parameters_.limits_enabled) {
      velocity_[i] = std::clamp(
        velocity_[i],
        -parameters_.max_velocity[i],
        parameters_.max_velocity[i]);
    }

    // 更新位移量
    displacement_[i] += velocity_[i] * dt;

    if (parameters_.limits_enabled) {
      const double limited_displacement = std::clamp(
        displacement_[i],
        -parameters_.max_displacement[i],
        parameters_.max_displacement[i]);

      if (limited_displacement != displacement_[i]) {
        displacement_[i] = limited_displacement;
        if (displacement_[i] * velocity_[i] > 0.0) {
          velocity_[i] = 0.0;
        }
      }
    }
  }

  return displacement_;
}

/**
 * @brief 重置导纳控制器内部状态
 * @details 将位移和速度修正量重置为零
 */
void AdmittanceCore::reset()
{
  displacement_.fill(0.0);
  velocity_.fill(0.0);
}

/**
 * @brief 获取当前的位移修正量
 * @return 位移修正量
 */
const Vector6 & AdmittanceCore::displacement() const
{
  return displacement_;
}

/**
 * @brief 获取当前的速度修正量
 * @return 速度修正量
 */
const Vector6 & AdmittanceCore::velocity() const
{
  return velocity_;
}

/**
 * @brief 应用死区处理
 * @param value 输入值
 * @param deadband 死区阈值
 * @return 处理后的值
 */
double AdmittanceCore::apply_deadband(
  const double value,
  const double deadband)
{
  if (std::abs(value) <= deadband) {
    return 0.0;
  }

  return value - std::copysign(deadband, value);
}

}  // namespace s622_dual_admittance
