/**
 * @file chirpz.hpp
 * @brief Header file for CPU based Chirp-z 3D FFT
 */

#ifndef CHIRPZ_HPP
#define CHIRPZ_HPP

#include <fftw3.h>

struct CONFIG{
  std::string path;
  std::string wisdomfile;
  std::string chirp_wisdomfile;
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

const unsigned next_second_power_of_two(unsigned x);

bool create_data(fftwf_complex *fftw_verify, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num_pts, const unsigned chirp_num_pts);

//bool fft_conv3D_cpu_verify(struct CONFIG& config, const float2 *sig, const float2 *filter, float2 *out);

#endif 