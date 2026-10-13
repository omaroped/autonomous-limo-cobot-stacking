cmake_minimum_required(VERSION 3.8)
project(limo_car)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

install(
  DIRECTORY config launch gazebo meshes rviz worlds scripts maps media
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/box_pose_estimator.py
  RENAME box_pose_estimator DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/base_joint_state_pub.py
  RENAME base_joint_state_pub DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS scripts/nav_pick_orchestrator.py
  RENAME nav_pick_orchestrator DESTINATION lib/${PROJECT_NAME})
# No RENAME: nav_pick_orchestrator imports it by module name ("from arm_interface import ArmResult").
install(PROGRAMS scripts/arm_interface.py DESTINATION lib/${PROJECT_NAME})

# ── Fiducial place pipeline ────────────────────────────────────────────────
install(PROGRAMS scripts/tag_dock_estimator.py
  RENAME tag_dock_estimator DESTINATION lib/${PROJECT_NAME})
# gen_apriltag.py is a one-shot generator (run once, not a ROS node)
install(PROGRAMS scripts/gen_apriltag.py
  RENAME gen_apriltag DESTINATION lib/${PROJECT_NAME})

# wait_for_lifecycle_node.py — blocks nav2_limo.launch.py's lifecycle manager
# start until map_server is actually ready, replacing a fixed 6 s guess that
# intermittently fired too early on a loaded machine.
install(PROGRAMS scripts/wait_for_lifecycle_node.py
  RENAME wait_for_lifecycle_node DESTINATION lib/${PROJECT_NAME})

ament_package()
