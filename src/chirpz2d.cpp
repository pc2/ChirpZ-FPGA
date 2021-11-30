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

// In-place 2d Transpose
void transpose2d(float2 *temp, const unsigned num){

  assert((temp != NULL));

  float2 *tmp = new float2[num * num];
  for(size_t i = 0; i < num; i++){
    for(size_t j = 0; j < num; j++){
      tmp[(i * num) + j].x = temp[(j * num) + i].x;
      tmp[(i * num) + j].y = temp[(j * num) + i].y;
    }
  }

  for(size_t i = 0; i < (num * num); i++){
    temp[i].x = tmp[i].x;
    temp[i].y = tmp[i].y;
  }
  free(tmp);
}

bool verify_chirp2d(vector<float2> inp, vector<float2> out, const unsigned num, const unsigned batch, const bool inverse){

  if(num <= 1){ throw("Bad number of points for verifying 2D FFT "); }

  const unsigned total_sz = batch * num * num;

  int *n = (int*)calloc(2 , sizeof(int));
  for(unsigned i = 0; i < 2; i++){
    n[i] = num;
  }
  
  int idist = num * num, odist = num * num;
  int istride = 1, ostride = 1; // contiguous in memory

  fftwf_complex *fftwf_verify = fftwf_alloc_complex(total_sz);

  fftwf_plan plan_fftwf;
  if(inverse){
    printf("\tInverse transform\n");
    plan_fftwf = fftwf_plan_many_dft(2, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_BACKWARD, FFTW_ESTIMATE);
  }
  else{
    plan_fftwf = fftwf_plan_many_dft(2, n, batch, &fftwf_verify[0], NULL, istride, idist, &fftwf_verify[0], NULL, ostride, odist, FFTW_FORWARD, FFTW_ESTIMATE);
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
  if(db > 90) // reducing SNR from 120 to 90
    status = true;
  else
    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", db, "FAILED");

  return status;
}

// Chirp Z implementation
void chirpz2d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inv, const unsigned batch){

  assert ( (inp != NULL) || (out != NULL));
  float2 *temp = new float2[num * num];
    
  for(unsigned iter = 0; iter < batch; iter++){

    unsigned index = (iter * num * num);

    // Row wise Chirp
    for(unsigned i = 0; i < num; i++){
      chirpz1d_cpu(&inp[index + (i * num)], &temp[i*num], num, inv);
    }
    // Transpose
    transpose2d(temp, num);

    // Column wise Chirp
    for(unsigned i = 0; i < num; i++){
      chirpz1d_cpu(&temp[i * num], &out[index + (i * num)], num, inv);
    }

    // Transpose
    transpose2d(&out[index], num);

  }
  delete[] temp;
}