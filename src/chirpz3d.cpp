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

void chirpz3d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inv, const unsigned batch){

  assert ( (inp != NULL) || (out != NULL));
  if(num <= 0){ throw("Bad number of points for 3D FFT "); }

  const unsigned chirp_num = next_second_power_of_two(num);
  printf("Chirp dim: %u x %u x %u \n", chirp_num, chirp_num, chirp_num);

  float2 *temp = new float2[num * num * num];
  
  for(unsigned iter = 0; iter < batch; iter++){

    unsigned index = (iter * num * num * num);
    // xy plane chirp
    for(size_t i = 0; i < num; i++){
      chirpz2d_cpu(&inp[index + (i * num * num)], &temp[i * num * num], num, inv, 1);
    }

    // so-called xz corner turn: xyz -> xzy
    transpose3d(temp, num);

    // yx plane chirp
    for(size_t i = 0; i < (num * num); i++){
      chirpz1d_cpu(&temp[i * num], &out[index + (i * num)], num, inv);
    }

    // reverse xz corner turn: xzy -> xyz
    transpose3d_rev(&out[index], num);
  }

  delete[] temp;
}

bool verify_chirp3d(vector<float2> inp, vector<float2> out, const unsigned num, const unsigned batch, const bool inverse){

  if(num <= 1){ throw("Bad number of points for verifying 3D FFT "); }

  const unsigned total_sz = batch * num * num * num;

  int *n = (int*)calloc(3 , sizeof(int));
  for(unsigned i = 0; i < 3; i++){
    n[i] = num;
  }

  int idist = num * num * num, odist = num *num * num;
  int istride = 1, ostride = 1; // contiguous in memory

  fftwf_complex *fftwf_verify = fftwf_alloc_complex(total_sz);

  fftwf_plan plan_fftwf;
  if(inverse){
    printf("\tInverse transform\n");
    plan_fftwf = fftwf_plan_many_dft(3, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan_fftwf = fftwf_plan_many_dft(3, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
  }

  for(unsigned i = 0; i < total_sz; i++){
    fftwf_verify[i][0] = inp[i].x;
    fftwf_verify[i][1] = inp[i].y;
  }
  fftwf_execute(plan_fftwf);

  float magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for(size_t i = 0; i < total_sz; i++) {
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
  for(unsigned i = 0; i < total_sz; i++){
    printf("%d: Impl: (%f, %f), FFTW: (%f, %f)\n", i, out[i].x, out[i].y, fftwf_verify[i][0], fftwf_verify[i][1]);
  }
  cout << endl << endl;
#endif

  fftwf_free(fftwf_verify);
  fftwf_destroy_plan(plan_fftwf);

  // Calculate SNR
  float db = 10 * log(mag_sum / noise_sum) / log(10.0);

  printf("SNR achieved: %f\n", db);
  bool status = false;
  if(db > 80) // reducing SNR from 120 to 80
    status = true;
  else
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");

  return status;
}