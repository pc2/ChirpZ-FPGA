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

typedef struct cpu_timing {
  double chirpz_t;      /**< Time for ChirpZ FFT */ 
  bool valid;           /**< True if valid execution */
} cpu_t;

cpu_t chirpz_cpu(struct CONFIG& config);

cpu_t chirpz_cpu_1d(float2 *inp, float2 *out, const struct CONFIG& config);

bool verify_chirp_1d(float2 *inp, float2 *out, const unsigned num);

#endif 