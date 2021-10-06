#include "gtest/gtest.h"
#include <iostream>
#include <fftw3.h>
#include "chirpz.hpp"
#include "helper.hpp"

TEST(ChirpZTest, ValidPowerOfTwo){
  EXPECT_EQ(next_second_power_of_two(61), 128);
  EXPECT_EQ(next_second_power_of_two(127), 256);
  EXPECT_EQ(next_second_power_of_two(251), 512);
  EXPECT_EQ(next_second_power_of_two(509), 1024);
}
