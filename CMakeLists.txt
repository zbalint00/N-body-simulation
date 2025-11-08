# Set the minimum required version of CMake.
# Modern CMake features are used, so a recent version is recommended.
cmake_minimum_required(VERSION 3.16)

if(NOT CMAKE_TOOLCHAIN_FILE AND DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "Using vcpkg toolchain from $ENV{VCPKG_ROOT}")
else()
  message(STATUS "ENV{VCPKG_ROOT} is not defined. Skipping using ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
endif()

#if(NOT CMAKE_TOOLCHAIN_FILE) # Optional: Only run if a toolchain isn't already specified externally
  include(cmake/bootstrap_vcpkg.cmake)
  #message(STATUS "Using cmake/bootstrap_vcpkg.cmake bootstrap script for locally installing vcpkg.")
#else()
  #message(STATUS "CMAKE_TOOLCHAIN_FILE is already set to: ${CMAKE_TOOLCHAIN_FILE}. Skipping vcpkg bootstrap script.")
#endif()

#==============================================================================
# USER-FACING OPTIONS
#==============================================================================
option(BUILD_WITH_OPENGL "Enable OpenGL rendering and visualization" OFF)

#==============================================================================
# CONDITIONALLY ENABLE VCPKG FEATURES
#==============================================================================
if(BUILD_WITH_OPENGL)
  # Tell vcpkg to install the 'opengl' feature from vcpkg.json
  set(VCPKG_MANIFEST_FEATURES "opengl" CACHE STRING "VCPKG features to install")
else()
  # Tell vcpkg to install no optional features (only core dependencies)
  set(VCPKG_MANIFEST_FEATURES "" CACHE STRING "VCPKG features to install")
endif()

# Define the project name, version, and language.
project(n_body_simulation VERSION 1.0 LANGUAGES CXX)

#==============================================================================
# COMPILE OPTIONS
#==============================================================================

# CXX_STANDARD_REQUIRED ensures that if the compiler doesn't support C++17,
# CMake will produce an error.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) # Discourage compiler-specific extensions

#==============================================================================
# DEPENDENCIES
#==============================================================================
# Find required external libraries.

find_package(OpenCL CONFIG REQUIRED)

#==============================================================================
# Interfaces, Common libraries, etc.
#==============================================================================

# Add the interface definition for OpenCL projects
include(cmake/OpenCLConfig.cmake)

if(BUILD_WITH_OPENGL)
  message(STATUS "OpenGL support is ENABLED.")
  
  # Add the interface definition for OpenGL projects
  include(cmake/OpenGLConfig.cmake)
else()
  message(STATUS "OpenGL support is DISABLED.")
  # No find_package calls = VCPKG doesn't install the feature.
endif()

#==============================================================================
# opencl-opengl interoperation projects
#==============================================================================
if(BUILD_WITH_OPENGL)

  #============================================================================
  # opencl_06_opengl_nbody
  #============================================================================

  add_subdirectory(src/opencl_06_opengl_nbody)
endif()