#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

static void copy_data_2d(float2 *inp, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  if(chirp_sig == NULL || chirp_filter == NULL || num <= 0 || chirp_num <= 0){
    throw("bad args for copying data in chirp 1d");
  }

  for(size_t i = 0; i < (chirp_num * chirp_num); i++){
    chirp_sig[i][0] = 0.0f;
    chirp_sig[i][1] = 0.0f;
    chirp_filter[i][0] = 0.0f;
    chirp_filter[i][1] = 0.0f;
  }

  for(size_t i = 0; i < num; i++){
    for(size_t j = 0; j < num; j++){
      unsigned index = (i*num) + j;
      unsigned chirp_index = (i*chirp_num) + j;

      chirp_sig[chirp_index][0] = inp[index].x;
      chirp_sig[chirp_index][1] = inp[index].y;
    }
  }
}

static void modulate_2d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < num; i++){
    for(size_t j = 0; j < num; j++){
      float x = cos(M_PI * j * j / num);
      float y = sin(M_PI * j * j / num);
      
      unsigned chirp_index = (i * chirp_num) + j;
      float a = chirp_sig[chirp_index][0]; 
      float b = chirp_sig[chirp_index][1]; 

      chirp_sig[chirp_index][0] = (x * b) + (y * a);
      chirp_sig[chirp_index][1] = (x * a) - (y * b);

      chirp_filter[chirp_index][0] = x;
      chirp_filter[chirp_index][1] = y;

      chirp_filter[(i*chirp_num) + (chirp_num -j)][0] = x;
      chirp_filter[(i*chirp_num) + (chirp_num -j)][1] = y;
    }
  }
}

static void point_mult_2d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned chirp_num){

  for(unsigned i = 0; i < (chirp_num * chirp_num); i++){
    float x, y;
    x = (chirp_sig[i][0] * chirp_filter[i][0]) - (chirp_sig[i][1] * chirp_filter[i][1]);
    y = (chirp_sig[i][0] * chirp_filter[i][1]) + (chirp_sig[i][1] * chirp_filter[i][0]);

    chirp_sig[i][0] = x;
    chirp_sig[i][1] = y;
  }
}

static void demodulate_2d(fftwf_complex *chirp_sig, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < chirp_num; i++){
    for(size_t j = 0; j < num; j++){
      float x = cos(M_PI * j * j / num);
      float y = sin(M_PI * j * j / num);

      unsigned chirp_index = (i * chirp_num) + j; 
      float a = chirp_sig[chirp_index][0];
      float b = chirp_sig[chirp_index][1];

      chirp_sig[chirp_index][1] = -1 * ((x * a) + (y * b)) / (chirp_num);
      chirp_sig[chirp_index][0] = ((x * b) - (y * a)) / (chirp_num);
    }
  }
}

bool verify_chirp_2d(float2 *inp, float2 *out, const unsigned num){

  cout << "Verifying Chirp2d" << endl;

  unsigned sz = num * num;
  fftwf_complex *fftwf_verify = fftwf_alloc_complex(sz);

  fftwf_plan plan_fftwf = fftwf_plan_dft_2d((int)num, (int)num, fftwf_verify, fftwf_verify, FFTW_FORWARD, FFTW_MEASURE);

  for(unsigned i = 0; i < sz; i++){
    fftwf_verify[i][0] = inp[i].x;
    fftwf_verify[i][1] = inp[i].y;

    printf("%d : inp - (%f %f) verify - (%f %f)\n", i, inp[i].x, inp[i].y, fftwf_verify[i][0], fftwf_verify[i][1]);
  }
  cout << endl;

  fftwf_execute(plan_fftwf);

  cout << "After execution: " << endl;
  for(unsigned i = 0; i < num*num; i++){
    printf("%d: Out: (%f, %f), FFTW: (%f, %f)\n", i, out[i].x, out[i].y, fftwf_verify[i][0], fftwf_verify[i][1]);
  }
  cout << endl;

  float magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for(size_t i = 0; i < num*num; i++) {
    magnitude = fftwf_verify[i][0] * fftwf_verify[i][0] + \
                      fftwf_verify[i][1] * fftwf_verify[i][1];
    noise = (fftwf_verify[i][0] - out[i].x) \
        * (fftwf_verify[i][0] - out[i].x) + 
        (fftwf_verify[i][1] - out[i].y) * (fftwf_verify[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
  }

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

// Chirp Z implementation
cpu_t chirpz_cpu_2d(float2 *inp, float2 *out, const struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};
  
  const unsigned num = config.num;
  //const unsigned num_sz = pow(num, config.dim);
  
  const unsigned chirp_num = next_second_power_of_two(num);
  const unsigned chirp_sz = pow(chirp_num, config.dim);

  cout << "-- FFT Dimensions: "<< num << "x" << num << endl;
  cout << "-- Chirp Dimensions: "<< chirp_num << "x" << chirp_num << endl;
  fftwf_complex *chirp_sig = fftwf_alloc_complex(chirp_sz);
  fftwf_complex *chirp_filter = fftwf_alloc_complex(chirp_sz);

  fftwf_plan plan_chirp_sig = fftwf_plan_dft_2d(chirp_num, chirp_num, chirp_sig, chirp_sig, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_chirp_filter = fftwf_plan_dft_2d(chirp_num, chirp_num, chirp_filter, chirp_filter, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_inv_chirp = fftwf_plan_dft_2d(chirp_num, chirp_num, chirp_sig, chirp_sig, FFTW_BACKWARD, FFTW_MEASURE);
  
  cout << "-- Creating data for FFT" << endl;
  copy_data_2d(inp, chirp_sig, chirp_filter, num, chirp_num);

  cout << "-- Modulating Input" << endl;
  modulate_2d(chirp_sig, chirp_filter, num, chirp_num);

  cout << "-- Executing Convolution" << endl;
  fftwf_execute(plan_chirp_filter);
  fftwf_execute(plan_chirp_sig);
  point_mult_2d(chirp_sig, chirp_filter, chirp_num);

  fftwf_execute(plan_inv_chirp);

  cout << "-- Demodulating Result" << endl;
  demodulate_2d(chirp_sig, num, chirp_num);

  for(unsigned i = 0; i < num; i++){
    for(unsigned j = 0; j < num; j++){
      unsigned index = (i * num) + j;
      unsigned chirp_index = (i * chirp_num) + j;

      out[index].x = chirp_sig[chirp_index][0];
      out[index].y = chirp_sig[chirp_index][1];
    }
  }

  timing_cpu.valid = true;

  cout << "-- Cleaning up" << endl;
  fftwf_free(chirp_sig);
  fftwf_free(chirp_filter);
  fftwf_destroy_plan(plan_chirp_sig);
  fftwf_destroy_plan(plan_chirp_filter);
  fftwf_destroy_plan(plan_inv_chirp);

  return timing_cpu;
}