/**
 * @file chirpz.hpp
 * @brief Header file for CPU based Chirp-z 3D FFT
 */

#ifndef CHIRPZ_HPP
#define CHIRPZ_HPP

#include <vector>
#include <fftw3.h>

//#include "../api/include/fftfpga/fftfpga.h"

#include "fftfpga/fftfpga.h"

struct CONFIG{
  std::string path;
  std::string wisdomfile;
  std::string chirp_wisdomfile;
  unsigned dim; 
  unsigned num; 
  unsigned iter;
  unsigned threads;
  unsigned batch;
  bool inv;
  bool noverify;
  bool cpuonly;
  bool use_svm;
  bool emulate;
};

void transpose2d(float2 *temp, const unsigned num);
void transpose3d(float2 *temp, const unsigned num);
void transpose3d_rev(float2 *temp, const unsigned num);

void chirpz1d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inverse);
bool verify_chirp1d(std::vector<float2> inp, std::vector<float2> out, const unsigned num, const unsigned batch, const bool inverse);

void chirpz2d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inverse, const unsigned batch);
bool verify_chirp2d(std::vector<float2> inp, std::vector<float2> out, const unsigned num, const unsigned batch, const bool inverse);

void chirpz3d_cpu(float2 *inp, float2 *out, const unsigned num, const bool inverse, const unsigned batch);
bool verify_chirp3d(std::vector<float2> inp, std::vector<float2> out, const unsigned num, const unsigned batch, const bool inverse);

#endif 