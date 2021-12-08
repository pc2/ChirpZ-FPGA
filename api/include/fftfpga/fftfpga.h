// Author: Arjun Ramaswami

/**
 * @file fftfpga.h
 * @brief Header file that provides APIs for OpenCL Host code
 */

#ifndef FFTFPGA_H
#define FFTFPGA_H

#include <stdbool.h>

/**
 * Single Precision Complex Floating Point Data Structure
 */
typedef struct {
  float x; /**< real value */
  float y; /**< imaginary value */
} float2;

/**
 * Double Precision Complex Floating Point Data Structure
 */
typedef struct {
  double x; /**< real value */
  double y; /**< imaginary value */
} double2;

/**
 * Record time in milliseconds of different FPGA runtime stages
 */
typedef struct fpga_timing {
  double pcie_read_t;     /**< Time to read from DDR to host using PCIe bus  */ 
  double pcie_write_t;    /**< Time to write from DDR to host using PCIe bus */ 
  double exec_t;          /**< Kernel execution time */
  double svm_copyin_t;    /**< Time to copy in data to SVM */
  double svm_copyout_t;   /**< Time to copy data out of SVM */ 
  bool valid;             /**< Represents true signifying valid execution */
} fpga_t;

#ifdef __cplusplus
extern "C" {
#endif
/** 
 * @brief Initialize FPGA
 * @param platform_name: name of the OpenCL platform
 * @param path         : path to binary
 * @param use_svm      : 1 if true 0 otherwise
 * @return 0 if successful 
          -1 Path to binary missing
          -2 Unable to find platform passed as argument
          -3 Unable to find devices for given OpenCL platform
          -4 Failed to create program, file not found in path
          -5 Device does not support required SVM
 */
extern int fpga_initialize(const char *platform_name, const char *path, const bool use_svm);

/** 
 * @brief Release FPGA Resources
 */
extern void fpga_final();

/** 
 * @brief Allocate memory of double precision complex floating points
 * @param sz  : size_t - size to allocate
 * @return void ptr or NULL
 */
extern void* fftfpga_complex_malloc(const size_t sz);

/** 
 * @brief Allocate memory of single precision complex floating points
 * @param sz  : size_t : size to allocate
 * @return void ptr or NULL
 */
extern void* fftfpgaf_complex_malloc(const size_t sz);

/**
 * @brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * @param  N    : integer pointer to size of FFT3d  
 * @param  inp  : float2 pointer to input data of size N
 * @param  out  : float2 pointer to output data of size N
 * @param  inv  : int toggle to activate backward FFT
 * @param  iter : number of iterations of the N point FFT
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_chirp1d(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

/**
 * @brief  compute an out-of-place single precision complex 2D-FFT using the BRAM of the FPGA
 * @param  N    : integer pointer to size of FFT2d  
 * @param  inp  : float2 pointer to input data of size [N * N]
 * @param  out  : float2 pointer to output data of size [N * N]
 * @param  inv  : int toggle to activate backward FFT
 * @param  how_many : number of 2D FFTs to computer, default 1
 * @return fpga_t : time taken in milliseconds for data transfers and execution
 */
extern fpga_t fftfpgaf_c2c_chirp2d_bram(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

extern fpga_t fftfpgaf_c2c_chirp2d_bram_v2(const unsigned N, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

/**
 * \brief  compute an out-of-place single precision complex 3D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in FFT1d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle for backward transforms
 * \param  batch : number of batched executions of 3D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_chirp3d(const unsigned num, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

fpga_t fftfpgaf_c2c_chirp3d_v2(const unsigned num, const float2 *inp, float2 *out, const bool inv, const unsigned batch);

#ifdef __cplusplus
}
#endif

#endif
