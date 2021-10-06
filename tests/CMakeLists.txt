project(chirpz-gtest)

add_executable(test_chirpz 
  test_chirpz.cpp 
  test_chirpz2d_cpu.cpp 
  test_chirpz3d_cpu.cpp 
  ${CMAKE_SOURCE_DIR}/src/chirpz1d.cpp 
  ${CMAKE_SOURCE_DIR}/src/chirpz2d.cpp 
  ${CMAKE_SOURCE_DIR}/src/chirpz3d.cpp 
  ${CMAKE_SOURCE_DIR}/src/helper.cpp
)

target_include_directories(test_chirpz 
  PRIVATE 
  ${chirpz-cpu_SOURCE_DIR} ${chirpz-cpu_SOURCE_DIR}/src/include
  ${IntelFPGAOpenCL_INCLUDE_DIRS} ${CMAKE_BINARY_DIR}/include  
  ${FFTW_INCLUDE_DIRS}
)

target_link_libraries(test_chirpz PRIVATE 
  gtest_main gtest gmock
  cxxopts OpenMP::OpenMP_CXX ${IntelFPGAOpenCL_LIBRARIES}
  fftw3 fftw3f ${FFTW_FLOAT_LIB} ${FFTW_FLOAT_OPENMP_LIB} 
)