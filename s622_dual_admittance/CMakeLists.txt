cmake_minimum_required(VERSION 3.8)
project(s622_dual_admittance)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(s622_dual_interfaces REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

add_executable(cartesian_pose_executor src/cartesian_pose_executor.cpp)
target_compile_features(cartesian_pose_executor PUBLIC cxx_std_17)
ament_target_dependencies(cartesian_pose_executor
  geometry_msgs moveit_core moveit_ros_planning rclcpp
  s622_dual_interfaces sensor_msgs std_srvs trajectory_msgs)
install(TARGETS cartesian_pose_executor DESTINATION lib/${PROJECT_NAME})

add_library(admittance_core
  src/admittance_core.cpp
)

target_compile_features(admittance_core PUBLIC cxx_std_17)

target_include_directories(admittance_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(admittance_node
  nodes/admittance_node.cpp
)

target_compile_features(admittance_node PUBLIC cxx_std_17)

ament_target_dependencies(admittance_node
  geometry_msgs
  s622_dual_interfaces
  rclcpp
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

target_link_libraries(admittance_node
  admittance_core
)

install(
  TARGETS admittance_core
  EXPORT export_admittance_core
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  TARGETS admittance_node
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

install(FILES README.md DESTINATION share/${PROJECT_NAME})

ament_export_targets(
  export_admittance_core
  HAS_LIBRARY_TARGET
)

if(BUILD_TESTING)
  add_executable(test_joint_position_limiter test/test_joint_position_limiter.cpp)
  target_compile_features(test_joint_position_limiter PUBLIC cxx_std_17)
  add_test(NAME test_joint_position_limiter COMMAND test_joint_position_limiter)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(
    test_admittance_core
    test/test_admittance_core.cpp
  )

  if(TARGET test_admittance_core)
    target_link_libraries(
      test_admittance_core
      admittance_core
    )
  endif()
endif()

ament_package()
