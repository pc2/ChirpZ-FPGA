/**
 * @file chirpz.hpp
 * @brief Header file for CPU based Chirp-z 3D FFT
 */

#ifndef CHIRPZ_HPP
#define CHIRPZ_HPP

#include <fftw3.h>

typedef struct {
  float x; /**< real value */
  float y; /**< imaginary value */
} float2;

struct CONFIG{
  std::string path;
  std::string wisdomfile;
  std::string chirp_wisdomfile;
  unsigned dim; 
  unsigned num; 
  unsigned iter;
  unsigned threads;
  unsigned batch;
  bool noverify;
  bool cpuonly;
  bool usesvm;
};

void transpose2d(float2 *temp, const unsigned num);
void transpose3d(float2 *temp, const unsigned num);
void transpose3d_rev(float2 *temp, const unsigned num);

void chirpz1d_cpu(float2 *inp, float2 *out, const unsigned num);
bool verify_chirp1d(float2 *inp, float2 *out, const unsigned num);

void chirpz2d_cpu(float2 *inp, float2 *out, const unsigned num);
bool verify_chirp2d(float2 *inp, float2 *out, const unsigned num);

void chirpz3d_cpu(float2 *inp, float2 *out, const unsigned num);
bool verify_chirp3d(float2 *inp, float2 *out, const unsigned num);

#endif 