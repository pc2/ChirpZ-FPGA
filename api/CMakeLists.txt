# Arjun Ramaswami
cmake_minimum_required(VERSION 3.18)
project(chirpz-api VERSION 1.0
            DESCRIPTION "APIs for ChirpZ FFTs using FPGAs"
            LANGUAGES C CXX)

##
# Generate host executable that is required to call OpenCL kernel bitstreams
# Target: host
##
add_library(chirpzfpga STATIC 
              ${PROJECT_SOURCE_DIR}/src/fftfpga.c 
              ${PROJECT_SOURCE_DIR}/src/chirp1d.c
              ${PROJECT_SOURCE_DIR}/src/chirp2d.c
              ${PROJECT_SOURCE_DIR}/src/chirp2d_v2.c
              ${PROJECT_SOURCE_DIR}/src/chirp3d.c
              ${PROJECT_SOURCE_DIR}/src/svm.c
              ${PROJECT_SOURCE_DIR}/src/opencl_utils.c
              ${PROJECT_SOURCE_DIR}/src/misc.c)

target_compile_options(chirpzfpga
    PRIVATE -Wall -Werror)

target_include_directories(chirpzfpga
    PRIVATE src 
    PUBLIC ${IntelFPGAOpenCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(chirpzfpga
    PUBLIC ${IntelFPGAOpenCL_LIBRARIES} m)