#include "gtest/gtest.h"
#include "chirpz.hpp"
#include "helper.hpp"

TEST(ChirpZTest, ValidPowerOfTwo){
  EXPECT_EQ(next_second_power_of_two(61), 128);
  EXPECT_EQ(next_second_power_of_two(127), 256);
  EXPECT_EQ(next_second_power_of_two(251), 512);
  EXPECT_EQ(next_second_power_of_two(509), 1024);
}

TEST(ChirpZTest, ValidInput){
  CONFIG test_config;
  test_config.num = 64;
  cpu_t cpu_timing = {0.0, false};

  cpu_timing = chirpz_cpu(test_config);
  EXPECT_TRUE(cpu_timing.valid);
}