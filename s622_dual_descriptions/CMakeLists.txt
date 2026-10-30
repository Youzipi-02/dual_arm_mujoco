cmake_minimum_required(VERSION 3.8)
project(s622_dual_descriptions)

find_package(ament_cmake REQUIRED)

# Preserve the relative paths used by scene.xml and the MJCF mesh directory.
install(
  DIRECTORY meshes robot_model config
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

ament_package()
