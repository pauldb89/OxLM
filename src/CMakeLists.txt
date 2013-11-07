# CMake Configuration File for OxLM Project.
cmake_minimum_required(VERSION 2.6)
project(OxLM)

#############################################
# Directories
#############################################

# Store all binaries in bin/ directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/../bin)
# Load additional CMake modules from cmake/ directory
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/../cmake)

# Include project path (src/)
include_directories(${PROJECT_SOURCE_DIR})

#############################################
# Libraries
#############################################

# Boost
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.45.0 COMPONENTS serialization program_options
  date_time REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# OpenMP
find_package(OpenMP REQUIRED)

#############################################
# Compiler Settings
#############################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")

add_subdirectory(cg)
