project(chirpz-gtest)

add_executable(test_chirpz test_chirpz.cpp)
target_link_libraries(test_chirpz PRIVATE gtest_main gtest gmock)
target_include_directories(test_chirpz PRIVATE ${CMAKE_SOURCE_DIR}/src)