#include <cmath>
#include <limits>
#include <stdexcept>

#include <gtest/gtest.h>

#include "s622_dual_admittance/admittance_core.hpp"

namespace
{

using s622_dual_admittance::AdmittanceCore;
using s622_dual_admittance::AdmittanceParameters;
using s622_dual_admittance::Vector6;

AdmittanceParameters make_parameters()
{
  AdmittanceParameters parameters;

  parameters.mass.fill(1.0);
  parameters.damping.fill(0.0);
  parameters.stiffness.fill(0.0);
  parameters.deadband.fill(0.0);
  parameters.max_velocity.fill(10.0);
  parameters.max_displacement.fill(10.0);
  parameters.enabled_axes.fill(true);

  return parameters;
}

TEST(AdmittanceCoreTest, ZeroWrenchProducesZeroDisplacement)
{
  AdmittanceCore core(make_parameters());

  const Vector6 zero{};
  const Vector6 displacement = core.update(zero, zero, 0.01);

  for (const double value : displacement) {
    EXPECT_DOUBLE_EQ(value, 0.0);
  }
}

TEST(AdmittanceCoreTest, PositiveYWrenchProducesPositiveYOffset)
{
  AdmittanceCore core(make_parameters());

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};

  external_wrench[1] = 1.0;

  const Vector6 displacement =
    core.update(external_wrench, desired_wrench, 0.01);

  // a = F/M = 1 m/s^2
  // v = a*dt = 0.01 m/s
  // x = v*dt = 0.0001 m
  EXPECT_NEAR(displacement[1], 0.0001, 1.0e-12);
  EXPECT_NEAR(core.velocity()[1], 0.01, 1.0e-12);

  for (std::size_t i = 0; i < 6; ++i) {
    if (i != 1) {
      EXPECT_DOUBLE_EQ(displacement[i], 0.0);
    }
  }
}

TEST(AdmittanceCoreTest, HighDampingConvergesWithoutAlternatingVelocity)
{
  auto parameters = make_parameters();
  parameters.mass[1] = 3.0;
  parameters.damping[1] = 2500.0;
  AdmittanceCore core(parameters);

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};
  external_wrench[1] = 10.0;

  double previous_velocity = 0.0;
  for (int cycle = 0; cycle < 100; ++cycle) {
    core.update(external_wrench, desired_wrench, 0.004);
    const double velocity = core.velocity()[1];
    EXPECT_GT(velocity, 0.0);
    EXPECT_GE(velocity, previous_velocity);
    previous_velocity = velocity;
  }
  EXPECT_NEAR(core.velocity()[1], 10.0 / 2500.0, 1.0e-10);

  external_wrench[1] = 0.0;
  for (int cycle = 0; cycle < 10; ++cycle) {
    core.update(external_wrench, desired_wrench, 0.004);
    const double velocity = core.velocity()[1];
    EXPECT_GE(velocity, 0.0);
    EXPECT_LE(velocity, previous_velocity);
    previous_velocity = velocity;
  }
}

TEST(AdmittanceCoreTest, DeadbandSuppressesSmallWrench)
{
  auto parameters = make_parameters();
  parameters.deadband[1] = 0.3;

  AdmittanceCore core(parameters);

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};

  external_wrench[1] = 0.2;

  const Vector6 displacement =
    core.update(external_wrench, desired_wrench, 0.01);

  EXPECT_DOUBLE_EQ(displacement[1], 0.0);
  EXPECT_DOUBLE_EQ(core.velocity()[1], 0.0);
}

TEST(AdmittanceCoreTest, DisabledAxisRemainsZero)
{
  auto parameters = make_parameters();
  parameters.enabled_axes[1] = false;

  AdmittanceCore core(parameters);

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};

  external_wrench[1] = 100.0;

  const Vector6 displacement =
    core.update(external_wrench, desired_wrench, 0.01);

  EXPECT_DOUBLE_EQ(displacement[1], 0.0);
  EXPECT_DOUBLE_EQ(core.velocity()[1], 0.0);
}

TEST(AdmittanceCoreTest, VelocityAndDisplacementAreLimited)
{
  auto parameters = make_parameters();

  parameters.max_velocity[1] = 0.1;
  parameters.max_displacement[1] = 0.001;

  AdmittanceCore core(parameters);

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};

  external_wrench[1] = 100.0;

  core.update(external_wrench, desired_wrench, 0.01);
  const Vector6 displacement =
    core.update(external_wrench, desired_wrench, 0.01);

  EXPECT_DOUBLE_EQ(displacement[1], 0.001);
  EXPECT_DOUBLE_EQ(core.velocity()[1], 0.0);
}

TEST(AdmittanceCoreTest, LimitsCanBeDisabled)
{
  auto parameters = make_parameters();
  parameters.limits_enabled = false;
  parameters.max_velocity[1] = 0.1;
  parameters.max_displacement[1] = 0.001;

  AdmittanceCore core(parameters);

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};
  external_wrench[1] = 100.0;

  core.update(external_wrench, desired_wrench, 0.01);
  const Vector6 displacement =
    core.update(external_wrench, desired_wrench, 0.01);

  EXPECT_NEAR(displacement[1], 0.03, 1.0e-12);
  EXPECT_NEAR(core.velocity()[1], 2.0, 1.0e-12);
}

TEST(AdmittanceCoreTest, ResetClearsInternalState)
{
  AdmittanceCore core(make_parameters());

  Vector6 external_wrench{};
  const Vector6 desired_wrench{};
  external_wrench[0] = 1.0;

  core.update(external_wrench, desired_wrench, 0.01);
  core.reset();

  for (std::size_t i = 0; i < 6; ++i) {
    EXPECT_DOUBLE_EQ(core.displacement()[i], 0.0);
    EXPECT_DOUBLE_EQ(core.velocity()[i], 0.0);
  }
}

TEST(AdmittanceCoreTest, RejectsInvalidControlPeriod)
{
  AdmittanceCore core(make_parameters());
  const Vector6 zero{};

  EXPECT_THROW(core.update(zero, zero, 0.0), std::invalid_argument);
  EXPECT_THROW(core.update(zero, zero, 0.021), std::invalid_argument);
}

TEST(AdmittanceCoreTest, RejectsNonFiniteParameter)
{
  auto parameters = make_parameters();
  parameters.deadband[0] =
    std::numeric_limits<double>::quiet_NaN();

  EXPECT_THROW(
    AdmittanceCore core(parameters),
    std::invalid_argument);
}

}  // namespace
