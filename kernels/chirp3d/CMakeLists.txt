# Author: Arjun Ramaswami
cmake_minimum_required(VERSION 3.18)

## 
# Call function to create custom build commands
# Generates targets:
#   - ${kernel_name}_emu: to generate emulation binary
#   - ${kernel_name}_rep: to generate report
#   - ${kernel_name}_syn: to generate synthesis binary
##
set(CL_PATH "${chirpz-kernels_SOURCE_DIR}/chirp3d")
set(kernels chirp3d chirp3d_v2)

include(${chirpz_SOURCE_DIR}/cmake/genKernelTargets.cmake)

if (INTELFPGAOPENCL_FOUND)
  gen_fft_targets(${kernels})
endif()