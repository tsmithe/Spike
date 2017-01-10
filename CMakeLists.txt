# Project setup
###############

cmake_minimum_required(VERSION 3.1 FATAL_ERROR) # only need 2.8.8, except for CMAKE_CXX_STANDARD
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)


project(Spike)

set (CMAKE_CXX_STANDARD 14)
# set (CMAKE_CXX_STANDARD_REQUIRED ON)

set(VERSION_MAJOR 1)
set(VERSION_MINOR 1)
set(VERSION_PATCH 0)
set(VERSION "${VERSION_MAJOR}.${VERSION_MINOR}.${VERSION_PATCH}")

#SET(CMAKE_VERBOSE_MAKEFILE ON)   #uncomment this line to increase CMake verbosity

#SET(BOOSTPATH "C:/Program\ Files\ (x86)/boost/boost_1_42")    #Boost path for other systems - customize as needed

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

# User options
##############

option(BUILD_WITH_CUDA
  "Enable CUDA backend"
  ON)

option(BUILD_WITH_OPENCL
  "Enable OpenCL backend"
  ON)

option(BUILD_WITH_VIENNACL
  "Enable Vienna backend"
  ON)

option(BUILD_EXAMPLES
  "Build examples"
  ON)

option(ENABLE_PLOTTING
  "Enable Plotting module -- requires mgl2"
  OFF)

option(BUILD_DOXYGEN_DOCS
  "Build the Doxygen-generated API docs"
  OFF)

option(BUILD_TESTING
  "Enable CTest tests"
  ON)

option(BUILD_SHARED_LIBS
  "Build shared libSpike (else static)"
  ON)

#cmake_dependent_option(ENABLE_DIST "Enable the the distribution tarball target" ON "BUILD_DOXYGEN_DOCS" OFF)
#mark_as_advanced(BUILD_DOXYGEN_DOCS ENABLE_DIST)

# Find prerequisites
####################

if(BUILD_DOXYGEN_DOCS)
  find_package(Doxygen REQUIRED)
endif()

if(BUILD_WITH_CUDA)
  find_package(CUDA REQUIRED)
  list(APPEND CUDA_NVCC_FLAGS "-std=c++11") # Shame that we can't use C++14!
  add_definitions(-DSPIKE_WITH_CUDA)
  include_directories(BEFORE SYSTEM "${CUDA_INCLUDE_DIRS}")
endif()

if(BUILD_WITH_OPENCL)
  find_package(OpenCL REQUIRED)
  # target_link_libraries(SpikeOpenCL ${OPENCL_LIBRARIES})
endif()

# find Eigen
set(EIGEN_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/Submodules/eigen")
find_path(EIGEN_INCLUDE_DIR Eigen/Dense)
if(NOT EIGEN_INCLUDE_DIR)
  message(SEND_ERROR "Failed to find Eigen; make sure you got the submodules!")
endif()
mark_as_advanced(EIGEN_INCLUDE_DIR)
include_directories(BEFORE SYSTEM "${EIGEN_INCLUDE_DIR}")

if(BUILD_WITH_VIENNACL)
  add_definitions(-DSPIKE_WITH_VIENNACL)
  include_directories(BEFORE SYSTEM "${PROJECT_SOURCE_DIR}/Submodules/viennacl-dev")
endif()

# Compile options
#################

include_directories(BEFORE "${PROJECT_SOURCE_DIR}")


# Subdirectories
################

add_subdirectory(Spike)

if(BUILD_EXAMPLES)
  add_subdirectory(Examples)
endif()

if(BUILD_TESTING)
  include(CTest)
  enable_testing()
  add_subdirectory(Tests)
endif()

# add_subdirectory(Doc)

# add_subdirectory(libspike)

# Install
#########

# install(DIRECTORY viennacl "${CMAKE_CURRENT_BINARY_DIR}/viennacl"
#    DESTINATION "${INSTALL_INCLUDE_DIR}" COMPONENT dev
#    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp")
