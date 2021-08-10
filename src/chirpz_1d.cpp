#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

static void copy_data_1d(float2 *inp, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  if(chirp_sig == NULL || chirp_filter == NULL || num <= 0 || chirp_num <= 0){
    throw("bad args for copying data in chirp 1d");
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

static void modulate_1d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < num; i++){
    float x = cos(M_PI * i * i / num);
    float y = sin(M_PI * i * i / num);
    float a = chirp_sig[i][0]; 
    float b = chirp_sig[i][1]; 

    chirp_sig[i][0] = (x * b) + (y * a);
    chirp_sig[i][1] = (x * a) - (y * b);

    chirp_filter[i][0] = x;
    chirp_filter[i][1] = y;

    chirp_filter[chirp_num -i][0] = x;
    chirp_filter[chirp_num -i][1] = y;
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

static void demodulate_1d(fftwf_complex *chirp_sig, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < num; i++){
    float x = cos(M_PI * i * i / num);
    float y = sin(M_PI * i * i / num);
    float a = chirp_sig[i][0];
    float b = chirp_sig[i][1];

    chirp_sig[i][1] = -1 * ((x * a) + (y * b)) / (chirp_num);
    chirp_sig[i][0] = ((x * b) - (y * a)) / (chirp_num);
  }
}

bool verify_chirp_1d(float2 *inp, float2 *out, const unsigned num){

  fftwf_complex *fftwf_verify = fftwf_alloc_complex(num);

  fftwf_plan plan_fftwf = fftwf_plan_dft_1d(num, fftwf_verify, fftwf_verify, FFTW_FORWARD, FFTW_MEASURE);

  for(unsigned i = 0; i < num; i++){
    fftwf_verify[i][0] = inp[i].x;
    fftwf_verify[i][1] = inp[i].y;
  }

  fftwf_execute(plan_fftwf);

  cout << endl;
  float magnitude = 0.0, noise = 0.0, mag_sum = 0.0, noise_sum = 0.0;
  for(size_t i = 0; i < num; i++) {
    magnitude = fftwf_verify[i][0] * fftwf_verify[i][0] + \
                      fftwf_verify[i][1] * fftwf_verify[i][1];
    noise = (fftwf_verify[i][0] - out[i].x) \
        * (fftwf_verify[i][0] - out[i].x) + 
        (fftwf_verify[i][1] - out[i].y) * (fftwf_verify[i][1] - out[i].y);

    mag_sum += magnitude;
    noise_sum += noise;
#ifndef NDEBUG
    printf("%zu : impl - (%e %e) fftw - (%e %e)\n", i, out[i].x, out[i].y, fftwf_verify[i][0], fftwf_verify[i][1]);
#endif            
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
cpu_t chirpz_cpu_1d(float2 *inp, float2 *out, const struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};
  
  const unsigned num = config.num;
  const unsigned chirp_num = next_second_power_of_two(num);

  cout << "-- FFT Dimensions: "<< num << endl;
  cout << "-- Chirp Dimensions: "<< chirp_num << endl;
  fftwf_complex *chirp_sig = fftwf_alloc_complex(chirp_num);
  fftwf_complex *chirp_filter = fftwf_alloc_complex(chirp_num);

  fftwf_plan plan_chirp_sig = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_chirp_filter = fftwf_plan_dft_1d(chirp_num, chirp_filter, chirp_filter, FFTW_FORWARD, FFTW_MEASURE);
  fftwf_plan plan_inv_chirp = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_BACKWARD, FFTW_MEASURE);
  
  cout << "-- Creating data for FFT" << endl;
  copy_data_1d(inp, chirp_sig, chirp_filter, num, chirp_num);

  cout << "-- Modulating Input" << endl;
  modulate_1d(chirp_sig, chirp_filter, num, chirp_num);

  cout << "-- Executing Convolution" << endl;
  fftwf_execute(plan_chirp_filter);
  fftwf_execute(plan_chirp_sig);
  point_mult_1d(chirp_sig, chirp_filter, chirp_num);

  fftwf_execute(plan_inv_chirp);

  cout << "-- Demodulating Result" << endl;
  demodulate_1d(chirp_sig, num, chirp_num);

  for(unsigned i = 0; i < num; i++){
    out[i].x = chirp_sig[i][0];
    out[i].y = chirp_sig[i][1];
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