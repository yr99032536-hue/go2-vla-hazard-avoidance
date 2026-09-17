cmake_minimum_required(VERSION 3.8)
project(go2_active_slam_interfaces)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PolicyObservationSnapshot.msg"
  "msg/ArmTrajectoryPoint.msg"
  "msg/ArmApplicationAck.msg"
  "action/RequestPolicyProposal.action"
  "action/ApplyArmTrajectory.action"
)

ament_python_install_package(go2_active_slam_interfaces_qos)
ament_python_install_package(go2_active_slam_protocol)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(
    test_idl_contract
    test/test_idl_contract.py
    APPEND_ENV PYTHONDONTWRITEBYTECODE=1
  )
  ament_add_pytest_test(
    test_serialization_goldens
    test/test_serialization_goldens.py
    APPEND_ENV PYTHONDONTWRITEBYTECODE=1
  )
  ament_add_pytest_test(
    test_generated_imports
    test/test_generated_imports.py
    APPEND_ENV P1A_REQUIRE_GENERATED=1 PYTHONDONTWRITEBYTECODE=1
  )
  ament_add_pytest_test(
    test_active_slam_protocol
    test/test_active_slam_protocol.py
    APPEND_ENV PYTHONDONTWRITEBYTECODE=1
  )
endif()

ament_export_dependencies(rosidl_default_runtime)
ament_package()
