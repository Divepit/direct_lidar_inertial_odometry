###########################################################
#                                                         #
# Copyright (c)                                           #
#                                                         #
# The Verifiable & Control-Theoretic Robotics (VECTR) Lab #
# University of California, Los Angeles                   #
#                                                         #
# Authors: Kenny J. Chen, Ryan Nemiroff, Brett T. Lopez   #
# Contact: {kennyjchen, ryguyn, btlopez}@ucla.edu         #
#                                                         #
###########################################################

cmake_minimum_required(VERSION 3.12.4)
project(direct_lidar_inertial_odometry)
find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)

find_package(PCL REQUIRED)
find_package( Eigen3 REQUIRED )
find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# Not all machines have <cpuid.h> available
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

rosidl_generate_interfaces(
    ${PROJECT_NAME}
        srv/SavePCD.srv
    DEPENDENCIES       
)
rosidl_get_typesupport_target(cpp_typesupport_target
  ${PROJECT_NAME} rosidl_typesupport_cpp)

include_directories(${PCL_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

include_directories(include)
include_directories(SYSTEM)

# --- Libraries you actually build ---
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
ament_target_dependencies(nanoflann)

add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} ${OpenMP_LIBS} nanoflann)
ament_target_dependencies(nano_gicp)

# --- Executables (unchanged) ---
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} ${OpenMP_LIBS} Threads::Threads nano_gicp)
ament_target_dependencies(dlio_odom_node "tf2_ros" "pcl_ros" "nav_msgs" "pcl_conversions" "rclcpp" "sensor_msgs" "geometry_msgs")

add_executable(dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES}  ${cpp_typesupport_target})
ament_target_dependencies(dlio_map_node "tf2_ros" "pcl_ros" "pcl_conversions" "rclcpp")

# --- Export & install headers ---
ament_export_include_directories(include)
install(DIRECTORY include/
        DESTINATION include)

# --- Install and export the actual library targets ---
# Create an export set so downstreams can link the targets in devel and after install
install(TARGETS nanoflann nano_gicp
        EXPORT export_${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# Export the targets and dependencies for consumers
ament_export_targets(export_${PROJECT_NAME})
ament_export_libraries(nano_gicp nanoflann)
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs pcl_ros)

# --- Install binaries (unchanged) ---
install(TARGETS dlio_odom_node dlio_map_node
        DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY cfg launch
        DESTINATION share/${PROJECT_NAME})

# --- Toolchain / flags cleanup ---
# You already set C++17 above; drop the conflicting -std=c++14
# remove: add_compile_options(-std=c++14)

# keep the rest (OpenMP detection etc.) as-is
ament_package()