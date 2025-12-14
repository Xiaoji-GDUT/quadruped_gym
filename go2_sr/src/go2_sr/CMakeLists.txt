cmake_minimum_required(VERSION 3.10)
project(go2_sr)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SYSTEM_TYPE ${CMAKE_SYSTEM_NAME})

set(USE_CMAKE OFF CACHE BOOL "Use CMake build system")
if (USE_CMAKE)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
endif()

if (NOT USE_CMAKE)
  add_compile_definitions(USE_ROS1)
else()
  message(WARNING "ROS_DISTRO not set, assuming non-ROS mode")
  add_compile_definitions(USE_CMAKE)
endif()

set(USE_MUJOCO OFF CACHE BOOL "Build with MuJoCo simulator support")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()
string(TOUPPER "${CMAKE_BUILD_TYPE}" UPPER_BUILD_TYPE)
if(UPPER_BUILD_TYPE STREQUAL "DEBUG")
  add_compile_options(-g)
endif()

# Get project root directory
get_filename_component(PROJECT_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
add_definitions(-DPOLICY_DIR="${PROJECT_ROOT_DIR}/policy")
add_definitions(-DBOOST_BIND_GLOBAL_PLACEHOLDERS)

if (NOT USE_CMAKE)
  find_package(catkin REQUIRED COMPONENTS
    controller_manager
    genmsg
    joint_state_controller
    robot_state_publisher
    roscpp
    gazebo_ros
    std_msgs
    tf
    geometry_msgs
    robot_msgs
    robot_joint_controller
    rospy
  )
  catkin_package(
    CATKIN_DEPENDS
    robot_joint_controller
    rospy
  )
  add_compile_options(${GAZEBO_CXX_FLAGS})
  find_package(gazebo REQUIRED)
endif()

# Inference Runtime
set(INFERENCE_RUNTIME_DIR "${PROJECT_ROOT_DIR}/library/inference_runtime")

# LibTorch Setup
set(USE_TORCH "OFF")
set(TORCH_LIBRARIES "")
set(LIBTORCH_DIR "${INFERENCE_RUNTIME_DIR}/libtorch")

if(EXISTS "${LIBTORCH_DIR}/include" AND EXISTS "${LIBTORCH_DIR}/lib")
    set(USE_TORCH "ON")
    add_compile_definitions(USE_TORCH)

    message(STATUS "Found LibTorch at: ${LIBTORCH_DIR}")

    # For Jetson with old CMake: set common Jetson CUDA architectures to avoid CUDA17 requirement
    if(CMAKE_VERSION VERSION_LESS "3.18")
        # Common Jetson architectures: 5.3(Nano), 6.2(TX2), 7.2(Xavier), 8.7(Orin)
        set(TORCH_CUDA_ARCH_LIST "5.3;6.2;7.2;8.7" CACHE STRING "CUDA architectures")
        set(ENV{TORCH_CUDA_ARCH_LIST} "5.3;6.2;7.2;8.7")
        message(STATUS "CMake ${CMAKE_VERSION} detected - using common Jetson CUDA architectures")
    endif()

    set(CMAKE_PREFIX_PATH "${LIBTORCH_DIR};${CMAKE_PREFIX_PATH}")
    find_package(Torch QUIET)

    if(Torch_FOUND)
        message(STATUS "LibTorch version: ${Torch_VERSION}")
        message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
    else()
        set(USE_TORCH "OFF")
        message(WARNING "LibTorch found but find_package(Torch) failed")
    endif()
else()
    message(WARNING "LibTorch not found. Please run: bash download_inference_runtime.sh libtorch")
endif()

# RPATH Configuration for inference libraries
if(USE_TORCH OR USE_ONNX)
    set(CMAKE_SKIP_BUILD_RPATH FALSE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH FALSE)

    set(PREBUILT_LIB_PATHS "")
    if(USE_TORCH)
        list(APPEND PREBUILT_LIB_PATHS "${LIBTORCH_DIR}/lib")
    endif()
    if(USE_ONNX)
        list(APPEND PREBUILT_LIB_PATHS "${ONNX_RUNTIME_DIR}/lib")
    endif()
    # OpenMP path will be added later after it's found on macOS
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${PREBUILT_LIB_PATHS}")

    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--disable-new-dtags")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--disable-new-dtags")
    endif()

    message(STATUS "Configured RPATH:")
    foreach(path ${PREBUILT_LIB_PATHS})
        message(STATUS "  - ${path}")
    endforeach()
endif()



find_package(TBB REQUIRED)
find_package(Threads REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Add Python library path to RPATH
if(Python3_LIBRARIES)
    get_filename_component(PYTHON_LIB_DIR "${Python3_LIBRARIES}" DIRECTORY)
    set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${PYTHON_LIB_DIR}")
    message(STATUS "Added Python library path to RPATH: ${PYTHON_LIB_DIR}")
endif()

set(YAML_CPP_LIBRARIES yaml-cpp)
link_directories(/usr/local/lib)
include_directories(${YAML_CPP_INCLUDE_DIR})

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    message(STATUS "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH_DIR "aarch64")
    set(UNITREE_LIB "libunitree_legged_sdk_arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
message(STATUS "Current system architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    set(ARCH_DIR "x86_64")
    set(UNITREE_LIB "libunitree_legged_sdk_amd64")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# unitree_legged_sdk
if(SYSTEM_TYPE STREQUAL "Linux")
add_library(unitree_legged_sdk SHARED IMPORTED)
set_target_properties(unitree_legged_sdk PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/include"
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/lib/${UNITREE_LIB}.so"
)
set(UNITREE_A1_LIBS Threads::Threads unitree_legged_sdk lcm)
if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
    file(GLOB GLOB_UNITREE_LEGGED_SDK "${CMAKE_CURRENT_SOURCE_DIR}/library/thirdparty/robot_sdk/unitree/unitree_legged_sdk/lib/${UNITREE_LIB}.so")
    install(FILES
        ${GLOB_UNITREE_LEGGED_SDK}
        DESTINATION lib/
    )
    endif()
endif()

# unitree_sdk2
if(SYSTEM_TYPE STREQUAL "Linux")
  set(BUILD_EXAMPLES OFF CACHE BOOL "Disable examples build for unitree_sdk2" FORCE)
  add_subdirectory(library/thirdparty/robot_sdk/unitree/unitree_sdk2)
endif()

if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "noetic")
        include_directories(${catkin_INCLUDE_DIRS})
    endif()
endif()

include_directories(
    include
    library/core/go2_sdk
    library/core/logger
    library/core/rl_sdk
    library/core/obs_buffer
    library/core/inference_runtime
    library/core/vector_math
    library/core/fsm
    library/core/loop

)

# library for inference runtime
add_library(inference_runtime library/core/inference_runtime/inference_runtime.cpp)
if(USE_TORCH)
    target_link_libraries(inference_runtime PUBLIC ${TORCH_LIBRARIES})
    target_compile_definitions(inference_runtime PRIVATE USE_TORCH)
    # Link OpenMP on macOS (required by LibTorch)
    if(SYSTEM_TYPE STREQUAL "Darwin" AND OPENMP_LIBRARIES)
        target_link_libraries(inference_runtime PUBLIC ${OPENMP_LIBRARIES})
    endif()
endif()
if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        install(TARGETS inference_runtime DESTINATION lib/${PROJECT_NAME})
    endif()
endif()

# library for rl sdk
add_library(rl_sdk library/core/rl_sdk/rl_sdk.cpp)
target_link_libraries(rl_sdk PUBLIC
    inference_runtime
    Python3::Python
    Python3::Module
    TBB::tbb
)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
    target_link_libraries(rl_sdk PUBLIC Python3::NumPy)
else()
    target_compile_definitions(rl_sdk PUBLIC WITHOUT_NUMPY)
endif()
if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        install(TARGETS rl_sdk DESTINATION lib/${PROJECT_NAME})
    endif()
endif()

# library for observation buffer
add_library(observation_buffer library/core/obs_buffer/obs_buffer.cpp)
if(NOT USE_CMAKE)
    if($ENV{ROS_DISTRO} MATCHES "foxy|humble")
        install(TARGETS observation_buffer DESTINATION lib/${PROJECT_NAME})
    endif()
endif()

# executable for rl sim
if(NOT USE_CMAKE)
  add_executable(rl_sim src/rl_sim.cpp)
  target_link_libraries(rl_sim
      rl_sdk
      observation_buffer
      ${YAML_CPP_LIBRARIES}
      Threads::Threads
  )
  target_link_libraries(rl_sim ${catkin_LIBRARIES})
endif()

# executable for actuator net
