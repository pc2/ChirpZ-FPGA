#include "gtest/gtest.h"
#include <iostream>
#include <fftw3.h>
#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

/**
 * \brief Transpose3d: test an xyz to xzy turn using 3d matrix with equal 2d matrices, which after transposition, has row-wise the same value.
 */
TEST(ChirpZ3D, Transpose3D){
  const unsigned num = 5;
  float2 *testinp = new float2[num * num * num];

  unsigned index = 0, value = 0;

  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      for(unsigned k = 0; k < num; k++){
        index = (i * num * num)+(j*num)+k;
        value = (float)((j*num)+k);

        testinp[index].x = value;
        testinp[index].y = value;
        //printf("%d: %f, %f\n", index, testinp[index].x, testinp[index].y);
      }
    }
  }

  transpose3d(testinp, num);
  
  for(unsigned i = 0; i < num*num; i++){
    for(unsigned j = 0; j < num; j++){
      index = (i* num)+ j;
      EXPECT_FLOAT_EQ(testinp[index].x, (float)(i));
      EXPECT_FLOAT_EQ(testinp[index].y, (float)(i));
    }
  }
    
  delete[] testinp;
}

/**
 * \brief Transpose3d_rev: routine to reverse after 3d FFT computation back to original form. The test executes on a transpose3d'ed routine and verifies with input array.
 */
TEST(ChirpZ3D, Transpose3D_rev){
  const unsigned num = 5;
  float2 *testinp = new float2[num * num * num];
  float2 *verifyinp = new float2[num * num * num];

  unsigned index = 0, value = 0;

  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      for(unsigned k = 0; k < num; k++){
        index = (i * num * num)+(j*num)+k;
        value = (float)((j*num)+k);

        testinp[index].x = value;
        testinp[index].y = value;
        verifyinp[index].x = value;
        verifyinp[index].y = value;
        //printf("%d: %f, %f\n", index, testinp[index].x, testinp[index].y);
      }
    }
  }

  transpose3d(testinp, num);
  
  transpose3d_rev(testinp, num);

  for(unsigned i = 0; i < num*num*num; i++){
    EXPECT_FLOAT_EQ(testinp[i].x, verifyinp[i].x);
    EXPECT_FLOAT_EQ(testinp[i].y, verifyinp[i].y);
    //printf("%d: (%f, %f) (%f, %f)\n", i, testinp[i].x, testinp[i].y, verifyinp[i].x, verifyinp[i].y);
  }
    
  delete[] verifyinp;
  delete[] testinp;
}