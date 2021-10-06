#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>
#include <assert.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

void transpose3d(float2 *inp, const unsigned num){

  assert((inp != NULL));
  if(num <= 1){ throw("Bad number of points for 3D Transpose "); }

  float2 *tmp = new float2[num * num * num];
  unsigned tmp_index, inp_index;

  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      for(unsigned k = 0; k < num; k++){
        tmp_index = (i * num * num) + (j * num) + k;
        inp_index = (k * num * num) + (i * num) + j;
        tmp[tmp_index].x = inp[inp_index].x;
        tmp[tmp_index].y = inp[inp_index].y;
      }
    }
  }

  for(unsigned i = 0; i < (num*num*num); i++){
    inp[i].x = tmp[i].x;
    inp[i].y = tmp[i].y;
  }

  delete[] tmp;
}

void transpose3d_rev(float2 *inp, const unsigned num){

  assert((inp != NULL));
  if(num <= 1){ throw("Bad number of points for 3D Transpose reversal"); }

  float2 *tmp = new float2[num * num * num];
  unsigned tmp_index, inp_index;
  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      for(unsigned k = 0; k < num; k++){
        inp_index = (i * num * num) + (j * num) + k;
        tmp_index = (k * num * num) + (i * num) + j;
        tmp[tmp_index].x = inp[inp_index].x;
        tmp[tmp_index].y = inp[inp_index].y;
      }
    }
  }

  for(unsigned i = 0; i < (num*num*num); i++){
    inp[i].x = tmp[i].x;
    inp[i].y = tmp[i].y;
  }

  delete[] tmp;
}

void chirpz3d_cpu(float2 *inp, float2 *out, const unsigned num){

  assert ( (inp != NULL) || (out != NULL));
  if(num <= 0){ throw("Bad number of points for 3D FFT "); }

  const unsigned chirp_num = next_second_power_of_two(num);
  printf("Chirp dim: %u x %u x %u \n", chirp_num, chirp_num, chirp_num);

  float2 *temp = new float2[num * num * num];
  
  // xy plane chirp
  for(size_t i = 0; i < num; i++){
    chirpz2d_cpu(&inp[i * num * num], &temp[i * num * num], num);
  }

  // so-called xz corner turn: xyz -> xzy
  transpose3d(temp, num);

  // yx plane chirp
  for(size_t i = 0; i < (num * num); i++){
    chirpz1d_cpu(&temp[i * num], &out[i * num], num);
  }

  // reverse xz corner turn: xzy -> xyz
  transpose3d_rev(out, num);

  delete[] temp;
}

bool verify_chirp3d(float2 *inp, float2 *out, const unsigned num){

  assert ( (inp != NULL) || (out != NULL));
  if(num <= 1){ throw("Bad number of points for verifying 3D FFT "); }

  unsigned sz = num * num * num;
  fftwf_complex *fftwf_verify = fftwf_alloc_complex(sz);

  fftwf_plan plan_fftwf = fftwf_plan_dft_3d((int)num, (int)num, (int)num, fftwf_verify, fftwf_verify, FFTW_FORWARD, FFTW_MEASURE);

  for(unsigned i = 0; i < sz; i++){
    fftwf_verify[i][0] = inp[i].x;
    fftwf_verify[i][1] = inp[i].y;
  }

  fftwf_execute(plan_fftwf);

  float magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for(size_t i = 0; i < sz; i++) {
    magnitude = fftwf_verify[i][0] * fftwf_verify[i][0] + \
                      fftwf_verify[i][1] * fftwf_verify[i][1];
    noise = (fftwf_verify[i][0] - out[i].x) \
        * (fftwf_verify[i][0] - out[i].x) + 
        (fftwf_verify[i][1] - out[i].y) * (fftwf_verify[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }

#ifndef NDEBUG
  cout << endl;
  for(unsigned i = 0; i < sz; i++){
    printf("%d: Impl: (%f, %f), FFTW: (%f, %f)\n", i, out[i].x, out[i].y, fftwf_verify[i][0], fftwf_verify[i][1]);
  }
  cout << endl << endl;
#endif

  fftwf_free(fftwf_verify);
  fftwf_destroy_plan(plan_fftwf);

  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);
  if(db > 120){
    return true;
  }
  else{
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");
    return false;
  }
}