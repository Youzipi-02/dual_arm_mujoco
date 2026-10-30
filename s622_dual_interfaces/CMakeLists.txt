cmake_minimum_required(VERSION 3.8)
project(s622_dual_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(vision_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "action/VisionPick.action"
  "action/ReleaseGrasp.action"
  "action/AcquireTargets.action"
  "action/PlanarSearch.action"
  "srv/ConfigureAdmittance.srv"
  "srv/PlanningLease.srv"
  "msg/AdmittanceStatus.msg"
  DEPENDENCIES geometry_msgs std_msgs vision_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
