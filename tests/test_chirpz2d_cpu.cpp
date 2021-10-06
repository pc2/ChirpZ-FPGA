#include "gtest/gtest.h"
#include <iostream>
#include <fftw3.h>
#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

/**
 * \brief Transpose2d: test an xy to yx turn 
 */
TEST(ChirpZ2D, Transpose2D){
  const unsigned num = 5;
  float2 *testinp = new float2[num * num];
  float2 *verifyinp = new float2[num * num];

  unsigned index = 0, value = 0;

  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      index = (i * num)+j;
      value = (float)(i);

      testinp[index].x = value;
      testinp[index].y = value;

      verifyinp[index].x = value;
      verifyinp[index].y = value;
    }
  }

  transpose2d(testinp, num);
  
  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      index = (i* num)+ j;
      EXPECT_FLOAT_EQ(testinp[index].x, (float)(j));
      EXPECT_FLOAT_EQ(testinp[index].y, (float)(j));
    }
  }
    
  transpose2d(testinp, num);

  for(unsigned i = 0; i < (num*num); i++){
    EXPECT_FLOAT_EQ(testinp[i].x, verifyinp[i].x);
    EXPECT_FLOAT_EQ(testinp[i].y, verifyinp[i].y);
  }

  delete[] verifyinp;
  delete[] testinp;
}
