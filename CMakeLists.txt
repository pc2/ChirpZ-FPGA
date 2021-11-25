# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.18)
project(chirpz 
          VERSION 1.0
          DESCRIPTION "Non-powers-of-2 3D FFT using Chirp-Z algorithm"
          LANGUAGES C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)

# build external dependencies
message("-- Building external dependencies")
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/extDep.cmake)

## find Intel FPGA SDK for OpenCL
find_package(IntelFPGAOpenCL REQUIRED)
message("-- IntelFPGAOpenCL found")

# Find FFTW
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
find_package(FFTW REQUIRED)
message("-- FFTW found")

find_package(OpenMP REQUIRED)

# build api 
message("-- Building the library")
add_subdirectory(api)

# build kernel
add_subdirectory(kernels)

# build source code
message("-- Building source code")
add_subdirectory(src)

# build tests
message("-- Building tests")
add_subdirectory(tests)
enable_testing()