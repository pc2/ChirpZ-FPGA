# External Dependencies
cmake_minimum_required(VERSION 3.18)
project(chirpz-cpu)

set(FFTW_PLAN "FFTW_ESTIMATE" CACHE STRING "FFTW Plan")
set_property(CACHE FFTW_PLAN PROPERTY STRINGS "FFTW_ESTIMATE" "FFTW_MEASURE" "FFTW_PATIENT" "FFTW_EXHAUSTIVE")

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/config.h.in"
  "${CMAKE_BINARY_DIR}/include/config.h"
  ESCAPE_QUOTES
)

add_executable(chirpz
  ${PROJECT_SOURCE_DIR}/main.cpp
  ${PROJECT_SOURCE_DIR}/chirpz.cpp
  ${PROJECT_SOURCE_DIR}/chirpz_1d.cpp
  ${PROJECT_SOURCE_DIR}/chirpz_2d.cpp
  ${PROJECT_SOURCE_DIR}/helper.cpp
)

target_compile_options(chirpz PRIVATE -Wall -Werror)
target_compile_features(chirpz PRIVATE cxx_std_17)

target_include_directories(chirpz 
  PRIVATE ${PROJECT_SOURCE_DIR} 
          ${PROJECT_SOURCE_DIR}/include
          ${IntelFPGAOpenCL_INCLUDE_DIRS}
          ${CMAKE_BINARY_DIR}/include 
          ${FFTW_INCLUDE_DIRS})

target_link_libraries(chirpz 
  PRIVATE cxxopts OpenMP::OpenMP_CXX  
    ${IntelFPGAOpenCL_LIBRARIES})

if(FFTW_FOUND)
  target_compile_definitions(chirpz PRIVATE USE_FFTW)
  target_link_libraries(chirpz PUBLIC fftw3 fftw3f ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB})
else()
  message(WARNING, "FFTW library not found. Cannot perform verification")
endif()