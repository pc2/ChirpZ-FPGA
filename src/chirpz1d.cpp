#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>
#include <vector>
#include <assert.h> 
#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

static void copy_data_1d(float2 *inp, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  assert ( (inp != NULL) | (chirp_sig != NULL) || (chirp_filter != NULL));

  if(num <= 0 || chirp_num <= 0 || chirp_num <= (2* num)){
    throw("Bad args for copying data into Chirp1d");
  }

  for(size_t i = 0; i < chirp_num; i++){
    chirp_sig[i][0] = 0.0f;
    chirp_sig[i][1] = 0.0f;
    chirp_filter[i][0] = 0.0f;
    chirp_filter[i][1] = 0.0f;
  }

  for(size_t i = 0; i < num; i++){
    chirp_sig[i][0] = inp[i].x;
    chirp_sig[i][1] = inp[i].y;
  }
}

static void modulate_1d(float2 *W, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  assert(W != NULL);

  for(size_t i = 0; i < num; i++){
    float x = W[i].x;
    float y = W[i].y;
    float a = chirp_sig[i][0]; 
    float b = chirp_sig[i][1]; 

    chirp_sig[i][0] = (x * a) - (y * b);
    chirp_sig[i][1] = (x * b) + (y * a);

    chirp_filter[i][0] = x;
    chirp_filter[i][1] = -1.0f * y;

    chirp_filter[chirp_num -i][0] = x;
    chirp_filter[chirp_num -i][1] = -1.0f * y;
  }
}

static void point_mult_1d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned chirp_num){

  for(unsigned i = 0; i < chirp_num; i++){
    float x, y;
    x = (chirp_sig[i][0] * chirp_filter[i][0]) - (chirp_sig[i][1] * chirp_filter[i][1]);
    y = (chirp_sig[i][0] * chirp_filter[i][1]) + (chirp_sig[i][1] * chirp_filter[i][0]);

    chirp_sig[i][0] = x;
    chirp_sig[i][1] = y;
  }
}

static void demodulate_1d(float2 *W, fftwf_complex *chirp_sig, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < num; i++){
    float x = W[i].x;
    float y = W[i].y;
    float a = chirp_sig[i][0];
    float b = chirp_sig[i][1];

    chirp_sig[i][0] = ((x * a) - (y * b));
    chirp_sig[i][1] = ((x * b) + (y * a));
  }
}

// Chirp Z implementation
void chirpz1d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inverse){

  const unsigned chirp_num = next_second_power_of_two(num);

  fftwf_complex *chirp_sig = fftwf_alloc_complex(chirp_num);
  fftwf_complex *chirp_filter = fftwf_alloc_complex(chirp_num);

  // Planning all FFTs  
  fftwf_plan plan_chirp_sig = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_chirp_filter = fftwf_plan_dft_1d(chirp_num, chirp_filter, chirp_filter, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_inv_chirp = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_BACKWARD, FFTW_MEASURE);
  
  copy_data_1d(inp, chirp_sig, chirp_filter, num, chirp_num);

  float sign = -1.0f;
  if(inverse)
    sign = 1.0f;

  float2 *W = new float2[num];
  for(unsigned i = 0; i < num; i++){
    W[i].x = cos(M_PIf32 * i * i / num);
    W[i].y = sign * sin(M_PIf32 * i * i / num);
  }

  modulate_1d(W, chirp_sig, chirp_filter, num, chirp_num);

  fftwf_execute(plan_chirp_filter);
  fftwf_execute(plan_chirp_sig);

  point_mult_1d(chirp_sig, chirp_filter, chirp_num);

  fftwf_execute(plan_inv_chirp);
  
  for(unsigned i = 0; i < chirp_num; i++){
    chirp_sig[i][0] = chirp_sig[i][0] / chirp_num;
    chirp_sig[i][1] = chirp_sig[i][1] / chirp_num;
  }
  
  demodulate_1d(W, chirp_sig, num, chirp_num);

  for(unsigned i = 0; i < num; i++){
    out[i].x = chirp_sig[i][0];
    out[i].y = chirp_sig[i][1];
  }

  delete[] W;
  fftwf_free(chirp_sig);
  fftwf_free(chirp_filter);
  fftwf_destroy_plan(plan_chirp_sig);
  fftwf_destroy_plan(plan_chirp_filter);
  fftwf_destroy_plan(plan_inv_chirp);
}

bool verify_chirp1d(vector<float2> inp, vector<float2> out, const unsigned num, const unsigned batch, const bool inverse){

  const unsigned total_sz = batch * num;

  int *n = (int*)calloc(1 , sizeof(int));
  for(unsigned i = 0; i < 1; i++){
    n[i] = num;
  }
  int idist = num, odist = num;
  int istride = 1, ostride = 1; // contiguous in memory

  fftwf_complex *fftwf_verify = fftwf_alloc_complex(total_sz);

  fftwf_plan plan_fftwf;
  if(inverse){
    printf("\tInverse transform\n");
    plan_fftwf = fftwf_plan_many_dft(1, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan_fftwf = fftwf_plan_many_dft(1, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
  }

  for(unsigned i = 0; i < total_sz; i++){
    fftwf_verify[i][0] = inp[i].x;
    fftwf_verify[i][1] = inp[i].y;
  }

  fftwf_execute(plan_fftwf);

  cout << endl;
  float magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for(unsigned i = 0; i < total_sz; i++) {
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

  printf("-- SNR achieved: %f db\n\n", db);
  bool status = false;
  if(db > 90) // reducing SNR from 120 to 90
    status = true;
  else
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");

  return status;
}