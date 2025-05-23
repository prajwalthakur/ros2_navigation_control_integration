cmake_minimum_required(VERSION 3.8)
project(custom_diff_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()


set(THIS_PACKAGE_INCLUDE_DEPENDS
  controller_interface
  generate_parameter_library
  geometry_msgs
  hardware_interface
  nav_msgs
  pluginlib
  rclcpp
  rclcpp_lifecycle
  rcpputils
  realtime_tools
  tf2
  tf2_msgs
)
find_package(ament_cmake REQUIRED)
find_package(backward_ros REQUIRED)

foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

generate_parameter_library(custom_diff_controller_parameters
  src/custom_diff_controller_parameters.yaml
)

add_library(custom_diff_controller SHARED
  src/custom_diff_controller.cpp
  src/odometry.cpp
  src/speed_limiter.cpp
)

target_compile_features(custom_diff_controller PUBLIC c_std_99 cxx_std_17)  # Require C99 and C++17
target_include_directories(custom_diff_controller PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include/custom_diff_controller>)

target_link_libraries(custom_diff_controller PUBLIC custom_diff_controller_parameters)
ament_target_dependencies( custom_diff_controller PUBLIC ${THIS_PACKAGE_INCLUDE_DEPENDS} )


# Causes the visibility macros to use dllexport rather than dllimport,
# which is appropriate when building the dll but not consuming it.
target_compile_definitions(custom_diff_controller PRIVATE "CUSTOM_DIFF_CONTROLLER_BUILDING_LIBRARY")
pluginlib_export_plugin_description_file(controller_interface custom_diff_controller_plugin.xml)


# if(BUILD_TESTING)
#   find_package(ament_cmake_gmock REQUIRED)
#   find_package(controller_manager REQUIRED)
#   find_package(ros2_control_test_assets REQUIRED)

#   ament_add_gmock(test_diff_drive_controller
#     test/test_diff_drive_controller.cpp)
#   target_link_libraries(test_diff_drive_controller
#     diff_drive_controller
#   )
#   ament_target_dependencies(test_diff_drive_controller
#     geometry_msgs
#     hardware_interface
#     nav_msgs
#     rclcpp
#     rclcpp_lifecycle
#     realtime_tools
#     tf2
#     tf2_msgs
#   )

#   add_rostest_with_parameters_gmock(test_load_diff_drive_controller
#     test/test_load_diff_drive_controller.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/test/config/test_diff_drive_controller.yaml
#   )
#   ament_target_dependencies(test_load_diff_drive_controller
#     controller_manager
#     ros2_control_test_assets
#   )
# endif()

install(
  DIRECTORY include/
  DESTINATION include/custom_diff_controller
)

install(
  TARGETS custom_diff_controller custom_diff_controller_parameters
  EXPORT export_${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

ament_export_targets(export_${PROJECT_NAME} HAS_LIBRARY_TARGET)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})

ament_package()
