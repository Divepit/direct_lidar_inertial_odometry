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

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(ENABLE_CLANG_TIDY "Run clang-tidy during the default build" OFF)

if(ENABLE_CLANG_TIDY)
  find_program(CLANG_TIDY_EXE NAMES clang-tidy)
  find_program(RUN_CLANG_TIDY_EXE NAMES run-clang-tidy run-clang-tidy.py)

  if(NOT CLANG_TIDY_EXE)
    message(STATUS "clang-tidy was not found. Skipping clang-tidy integration.")
    set(ENABLE_CLANG_TIDY OFF)
  elseif(NOT RUN_CLANG_TIDY_EXE)
    message(STATUS "run-clang-tidy was not found. Skipping clang-tidy integration.")
    set(ENABLE_CLANG_TIDY OFF)
  endif()
endif()

find_package(rosidl_default_generators REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(ament_index_cpp REQUIRED)

find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED )

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)

add_library(dlio_openmp INTERFACE)
target_link_libraries(dlio_openmp INTERFACE Threads::Threads)

if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  execute_process(
    COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=libgomp.so
    OUTPUT_VARIABLE DLIO_GOMP_LIBRARY_FROM_COMPILER
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)

  if(DLIO_GOMP_LIBRARY_FROM_COMPILER
      AND NOT DLIO_GOMP_LIBRARY_FROM_COMPILER STREQUAL "libgomp.so"
      AND EXISTS "${DLIO_GOMP_LIBRARY_FROM_COMPILER}")
    get_filename_component(DLIO_GOMP_LIBRARY "${DLIO_GOMP_LIBRARY_FROM_COMPILER}" REALPATH)
  else()
    find_library(DLIO_GOMP_LIBRARY NAMES gomp)
  endif()

  if(NOT DLIO_GOMP_LIBRARY)
    message(FATAL_ERROR "libgomp was not found. Install libgomp or build direct_lidar_inertial_odometry with GCC.")
  endif()

  include(CheckCXXSourceCompiles)
  set(CMAKE_REQUIRED_FLAGS_SAVE "${CMAKE_REQUIRED_FLAGS}")
  set(CMAKE_REQUIRED_LIBRARIES_SAVE "${CMAKE_REQUIRED_LIBRARIES}")
  set(CMAKE_REQUIRED_FLAGS "-pthread -fopenmp=libgomp")
  set(CMAKE_REQUIRED_LIBRARIES "${DLIO_GOMP_LIBRARY}")
  check_cxx_source_compiles(
    "#include <omp.h>
     int main() { return omp_get_max_threads() > 0 ? 0 : 1; }"
    DLIO_CLANG_LIBGOMP_OPENMP_WORKS)
  set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS_SAVE}")
  set(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES_SAVE}")

  if(NOT DLIO_CLANG_LIBGOMP_OPENMP_WORKS)
    message(FATAL_ERROR "Clang could not compile/link OpenMP with libgomp. Check that clang can use GCC's OpenMP headers and libgomp.")
  endif()

  # Use GNU OpenMP with Clang to avoid the Eigen/libomp conflict seen in this package.
  target_compile_options(dlio_openmp INTERFACE -pthread -fopenmp=libgomp)
  target_link_libraries(dlio_openmp INTERFACE ${DLIO_GOMP_LIBRARY})
else()
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(dlio_openmp INTERFACE OpenMP::OpenMP_CXX)
endif()

add_compile_options(-O3 -mtune=native -march=native -funroll-loops -DNDEBUG) #-funsafe-math-optimizations -funsafe-loop-optimizations
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options(-fopt-info-vec-optimized)
endif()

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

# NanoFLANN
add_library(nanoflann STATIC
  src/nano_gicp/nanoflann.cc
)
target_link_libraries(nanoflann ${PCL_LIBRARIES})
ament_target_dependencies(nanoflann)

# NanoGICP
add_library(nano_gicp STATIC
  src/nano_gicp/lsq_registration.cc
  src/nano_gicp/nano_gicp.cc
)
target_link_libraries(nano_gicp ${PCL_LIBRARIES} dlio_openmp nanoflann)
ament_target_dependencies(nano_gicp)

# Dynamic object filter
add_library(dlio_dynamic_filter STATIC
  src/dlio/m_detector_filter.cc
)
target_link_libraries(dlio_dynamic_filter ${PCL_LIBRARIES})
ament_target_dependencies(dlio_dynamic_filter)

# Odometry Node
add_executable(dlio_odom_node src/dlio/odom_node.cc src/dlio/odom.cc)
target_link_libraries(dlio_odom_node ${PCL_LIBRARIES} dlio_openmp nano_gicp dlio_dynamic_filter)
ament_target_dependencies(dlio_odom_node "tf2_ros" "pcl_ros" "nav_msgs" "pcl_conversions" "rclcpp" "sensor_msgs" "geometry_msgs" "visualization_msgs" "std_srvs" "ament_index_cpp")

# Mapping Node
add_executable (dlio_map_node src/dlio/map_node.cc src/dlio/map.cc)
target_link_libraries(dlio_map_node ${PCL_LIBRARIES}  ${cpp_typesupport_target})
ament_target_dependencies(dlio_map_node "tf2_ros" "pcl_ros" "pcl_conversions" "rclcpp" "nav_msgs" "std_srvs")

if(ENABLE_CLANG_TIDY)
  file(GLOB_RECURSE DLIO_CLANG_TIDY_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc"
  )

  add_custom_target(clang_tidy ALL
    COMMAND "${RUN_CLANG_TIDY_EXE}"
      -quiet
      -clang-tidy-binary "${CLANG_TIDY_EXE}"
      -p "${CMAKE_BINARY_DIR}"
      -extra-arg-before=--driver-mode=g++
      ${DLIO_CLANG_TIDY_SOURCES}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    COMMENT "Running standalone clang-tidy analysis"
    USES_TERMINAL
    VERBATIM
  )

  if(TARGET ${PROJECT_NAME}__cpp)
    add_dependencies(clang_tidy ${PROJECT_NAME}__cpp)
  endif()
endif()

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME} nano_gicp nanoflann)
ament_export_dependencies(rclcpp std_msgs sensor_msgs geometry_msgs pcl_ros)
ament_package()

# Binaries
install( TARGETS dlio_odom_node dlio_map_node
         DESTINATION lib/${PROJECT_NAME} 
)
install( DIRECTORY cfg launch
         DESTINATION share/${PROJECT_NAME} 
)
install( PROGRAMS scripts/plot_run_stats.py
         DESTINATION share/${PROJECT_NAME}/scripts
)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_m_detector_filter
    test/test_m_detector_filter.cc
  )
  target_link_libraries(test_m_detector_filter dlio_dynamic_filter ${PCL_LIBRARIES})
  target_include_directories(test_m_detector_filter PRIVATE include)
  target_compile_definitions(test_m_detector_filter PRIVATE DLIO_ENABLE_TEST_ACCESS)

  ament_add_gtest(test_imu_unit_scaler
    test/test_imu_unit_scaler.cc
  )
  target_include_directories(test_imu_unit_scaler PRIVATE include)

  ament_add_gtest(test_scan_time_bounds
    test/test_scan_time_bounds.cc
  )
  target_include_directories(test_scan_time_bounds PRIVATE include)
  ament_target_dependencies(test_scan_time_bounds sensor_msgs)

  add_test(
    NAME test_run_stats_plotter
    COMMAND bash
            ${CMAKE_CURRENT_SOURCE_DIR}/test/test_run_stats_plotter.sh
            ${CMAKE_CURRENT_SOURCE_DIR}/scripts/plot_run_stats.py
            ${CMAKE_CURRENT_BINARY_DIR}/run_stats_plotter_test
  )
endif()

link_directories(${PCL_LIBRARY_DIRS})
file(WRITE ${CMAKE_BINARY_DIR}/test_cpuid.cpp "#include <cpuid.h>")
try_compile(HAS_CPUID ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
file(REMOVE ${CMAKE_BINARY_DIR}/test_cpuid.cpp)
if(HAS_CPUID)
  add_compile_definitions(HAS_CPUID)
endif()
