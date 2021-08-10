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

/*
TEST(ChirpZTest, ValidCreateData){
  unsigned num_pts = 31;
  unsigned chirp_num_pts = next_second_power_of_two(num_pts);

  fftwf_complex *fftwf_a = fftwf_alloc_complex(num_pts);
  fftwf_complex *fftwf_b = fftwf_alloc_complex(chirp_num_pts);
  fftwf_complex *fftwf_c = fftwf_alloc_complex(chirp_num_pts);

  bool status = create_data(fftwf_a, fftwf_b, fftwf_c, num_pts, chirp_num_pts);
  EXPECT_TRUE(status);

  bool flag = true;
  for(size_t i = 0; i < num_pts; i++){
    if( (fftwf_a[i][0] != fftwf_b[i][0]) || (fftwf_a[i][1] != fftwf_b[i][1])){
      flag = false; 
      break;
    }
  }

  EXPECT_TRUE(flag);

  fftwf_free(fftwf_a);
  fftwf_free(fftwf_b);
  fftwf_free(fftwf_c);
}
*/