// Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdbool.h>
#define CL_VERSION_2_0
#include <CL/cl_ext_intelfpga.h> // to disable interleaving & transfer data to specific banks - CL_CHANNEL_1_INTELFPGA
#include "CL/opencl.h"

#include "fpga_state.h"
#include "fftfpga/fftfpga.h"
#include "svm.h"
#include "opencl_utils.h"
#include "misc.h"

#define NO_MODULATE 0
#define MODULATE 1

/**
 * \brief  compute an out-of-place single precision complex 1D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in FFT1d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle for backward transforms
 * \param  batch : number of batched executions of 1D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_chirp1d(const unsigned num, const float2 *inp, float2 *out, const bool inv, const unsigned batch){

  fpga_t fft_time = {0.0, 0.0, 0.0, 0.0, 0.0, 0};

  if(inp == NULL || out == NULL || (num < 8))
    return fft_time;

  cl_int status = 0;
  unsigned next_num = next_power_of_two(num);
  unsigned chirp_num = next_second_power_of_two(num);

  printf("\tNumber of points: %u\n\tNext Power of 2: %u\n\tChirp Power of 2: %u\n", num, next_num, chirp_num);
  
  // Zero array to initialize device buffer for W
  float2 temp[chirp_num];
  for(unsigned i = 0; i < chirp_num; i++){
    temp[i].x = 0.0f;
    temp[i].y = 0.0f;
  }

  // Reorder the incoming signal with zeroes to the nearest power of 2
  float2 *temp_in = malloc(sizeof(float2) * next_num * batch);
  for(unsigned i = 0; i < batch; i++){
    for(unsigned j = 0; j < next_num; j++){
      temp_in[(i*next_num)+j].x = (j < num) ? inp[(i * num) + j].x : 0.0f;
      temp_in[(i*next_num)+j].y = (j < num) ? inp[(i * num) + j].y : 0.0f;
    }
  }

  // Zero filled output buffer padded to the nearest power of 2 
  float2 *temp_out = malloc(sizeof(float2) * next_num * batch);
  for(unsigned i = 0; i < next_num * batch; i++){
    temp_out[i].x = 0.0f;
    temp_out[i].y = 0.0f;
  }

  /*
  *  Generate W array: by passing the next highest power-of-2 instead of
  *    the power-of-2 >= 2N-1, we aim to reduce the number of cycles of latency
  *    in some inputs
  *  E.g.: if num is 31 -> 32 but the required power-of-2 is 64.
  *  The buffer is of length 64 and initialized to 0s.
  */
  cl_mem d_bufW = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * next_num, NULL, &status);
  checkError(status, "Failed to allocate W buffer\n");

  cl_mem d_Filter = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_2_INTELFPGA, sizeof(float2) * chirp_num, NULL, &status);
  checkError(status, "Failed to allocate Filter buffer\n");

  cl_mem d_Signal = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * next_num * batch, NULL, &status);
  checkError(status, "Failed to allocate Signal buffer\n");

  cl_mem d_Out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * next_num * batch, NULL, &status);
  checkError(status, "Failed to allocate Signal buffer\n");

  const unsigned sign = (inv == true) ? 1 : 0;
  printf("\tInverse: %u\n\n", sign);

  cl_kernel kernel_gen_W = clCreateKernel(program, "gen_W_Filter", &status);
  checkError(status, "Failed to create gen_W kernel");
  status = clSetKernelArg(kernel_gen_W, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel gen_W arg 0");
  status = clSetKernelArg(kernel_gen_W, 1, sizeof(cl_mem), (void *)&d_Filter);
  checkError(status, "Failed to set kernel gen_W arg 1");
  status = clSetKernelArg(kernel_gen_W, 2, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel gen_W arg 2");
  status = clSetKernelArg(kernel_gen_W, 3, sizeof(cl_uint), (void *)&sign);
  checkError(status, "Failed to set kernel gen_W arg 3");

  queue_setup();

  // Initialize device buffer with zeroes for W
  status = clEnqueueWriteBuffer(queue1, d_bufW, CL_TRUE, 0, sizeof(float2) * next_num, temp, 0, NULL, NULL);
  checkError(status, "failed to finish writing zeroes to device buffer d_bufW");
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");

  status = clEnqueueWriteBuffer(queue1, d_Filter, CL_TRUE, 0, sizeof(float2) * chirp_num, temp, 0, NULL, NULL);
  checkError(status, "failed to finish writing zeroes to device buffer d_Filter");
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");

  /*
  *  Transforming the filter array
  */
  unsigned count = 1;
  cl_kernel kernel_fetch = clCreateKernel(program, "fetch", &status);
  checkError(status, "Failed to create fetch kernel");
  status = clSetKernelArg(kernel_fetch, 0, sizeof(cl_mem), (void *)&d_Filter);
  checkError(status, "Failed to set kernel fetch arg 0");
  status = clSetKernelArg(kernel_fetch, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel fetch arg 1"); 
  unsigned modulate = NO_MODULATE;
  status = clSetKernelArg(kernel_fetch, 2, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel fetch arg 2"); 
  status = clSetKernelArg(kernel_fetch, 3, sizeof(cl_uint), (void *)&count);
  checkError(status, "Failed to set kernel fetch arg 3");

  cl_kernel kernel_mod = clCreateKernel(program, "modulate", &status);
  checkError(status, "Failed to create modulate kernel");
  status = clSetKernelArg(kernel_mod, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel modulate arg 0");
  status = clSetKernelArg(kernel_mod, 1, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel modulate arg 1"); 
  status = clSetKernelArg(kernel_mod, 2, sizeof(cl_uint), (void *)&count);
  checkError(status, "Failed to set kernel modulate arg 2"); 

  unsigned total = count + batch;
  cl_kernel kernel_fft1da = clCreateKernel(program, "fft1da", &status);
  checkError(status, "Failed to create fft1da kernel");
  status = clSetKernelArg(kernel_fft1da, 0, sizeof(cl_uint), (void *)&total);
  checkError(status, "Failed to set kernel fft1d arg 0"); 
  int inverse = 0;
  status = clSetKernelArg(kernel_fft1da, 1, sizeof(cl_int), (void *)&inverse);
  checkError(status, "Failed to set kernel fft1d arg 1"); 

  // Filter is stored in BRAM, then multiplied with count * signals 
  cl_kernel kernel_mult = clCreateKernel(program, "multiplication", &status);
  checkError(status, "Failed to create multiplication kernel");
  status = clSetKernelArg(kernel_mult, 0, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel mult arg 0"); 
 
  // Arguments for Signal Transformation
  int inverse_sig = 1;
  cl_kernel kernel_fftinv1d = clCreateKernel(program, "fft1db", &status);
  checkError(status, "Failed to create fft1db kernel");
  status = clSetKernelArg(kernel_fftinv1d, 0, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel Inv FFT1D arg 0"); 
  status = clSetKernelArg(kernel_fftinv1d, 1, sizeof(cl_int), (void *)&inverse_sig);
  checkError(status, "Failed to set kernel Inv FFT1D arg 1");

  float factor = 1.0f / (float)chirp_num;

  cl_kernel kernel_scale = clCreateKernel(program, "scale", &status);
  status = clSetKernelArg(kernel_scale, 0, sizeof(cl_float), (void *)&factor);
  checkError(status, "Failed to set kernel scale arg 0"); 
  status = clSetKernelArg(kernel_scale, 1, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel scale arg 1"); 

  cl_kernel kernel_demod = clCreateKernel(program, "demodulate", &status);
  status = clSetKernelArg(kernel_demod, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel modulate arg 0");
  status = clSetKernelArg(kernel_demod, 1, sizeof(cl_mem), (void *)&d_Out);
  checkError(status, "Failed to set kernel modulate arg 1");
  status = clSetKernelArg(kernel_demod, 2, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel modulate arg 2"); 
  status = clSetKernelArg(kernel_demod, 3, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel modulate arg 3"); 

  // Overlap transferring signal with W generation and Filter transformation
  cl_event startFilter_event;
  cl_event startSignal_event, stopSignal_event;
  cl_event writeBuf_event, genW_event;
  status = clEnqueueWriteBuffer(queue1, d_Signal, CL_FALSE, 0, sizeof(float2) * next_num * batch, temp_in, 0, NULL, &writeBuf_event);
  checkError(status, "failed to finish writing inp data to device buffer d_Signal");

  status = clEnqueueTask(queue2, kernel_gen_W, 0, NULL, &genW_event);
  checkError(status, "Failed to launch gen_W kernel");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  /*
  *  Transform Filter and store in bitreversed order in multiplication kernel
  */
  // kernel_mult and kernel_fft1da continues to execute also the signal
  status = clEnqueueTask(queue7, kernel_demod, 0, NULL, &stopSignal_event);
  checkError(status, "Failed to launch demod kernel");
  status = clEnqueueTask(queue5, kernel_mult, 0, NULL, NULL);
  checkError(status, "Failed to launch multiplication kernel");
  status = clEnqueueTask(queue4, kernel_fft1da, 0, NULL, NULL);
  checkError(status, "Failed to launch fft1d kernel");
  status = clEnqueueTask(queue3, kernel_mod, 0, NULL, NULL);
  checkError(status, "Failed to launch modulate kernel");
  status = clEnqueueTask(queue2, kernel_fetch, 0, NULL, &startFilter_event);
  checkError(status, "Failed to launch fetch kernel");

  // fft1d and mult continue to process the signal data also
  status = clFinish(queue3);
  checkError(status, "Failed to finish queue3");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");

  clWaitForEvents(1, &writeBuf_event);
  status = clFinish(queue1); // queue1 writes signal to global memory
  checkError(status, "Failed to finish queue1");

  /*
  * Transform Signal
  */
  count = 1;
  modulate = MODULATE;
  status = clSetKernelArg(kernel_fetch, 0, sizeof(cl_mem), (void *)&d_Signal);
  checkError(status, "Failed to set kernel fetch arg 0");
  status = clSetKernelArg(kernel_fetch, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel fetch arg 1"); 
  status = clSetKernelArg(kernel_fetch, 2, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel fetch arg 2"); 
  status = clSetKernelArg(kernel_fetch, 3, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel fetch arg 3");

  status = clSetKernelArg(kernel_mod, 1, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel modulate arg 1"); 
  status = clSetKernelArg(kernel_mod, 2, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel modulate arg 2"); 

  // Queues 4,5,7 are occupied by FFT1da, mult and demod kernels respectively
  status = clEnqueueTask(queue6, kernel_scale, 0, NULL, NULL);
  checkError(status, "Failed to launch scale kernel");
  status = clEnqueueTask(queue3, kernel_fftinv1d, 0, NULL, NULL);
  checkError(status, "Failed to launch multiplication kernel");
  status = clEnqueueTask(queue2, kernel_mod, 0, NULL, NULL);
  checkError(status, "Failed to launch modulate kernel");
  status = clEnqueueTask(queue1, kernel_fetch, 0, NULL, &startSignal_event);
  checkError(status, "Failed to launch ifft kernel");
  
  status = clFinish(queue7);
  checkError(status, "Failed to finish queue7");
  status = clFinish(queue6);
  checkError(status, "Failed to finish queue6");
  status = clFinish(queue5);
  checkError(status, "Failed to finish queue5");
  status = clFinish(queue4);
  checkError(status, "Failed to finish queue4");
  status = clFinish(queue3);
  checkError(status, "Failed to finish queue3");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  status = clFinish(queue1);
  checkError(status, "Failed to finish queue1");
  
  cl_event readBuf_event;
  printf("-- Transferring output back to host\n");
  status = clEnqueueReadBuffer(queue1, d_Out, CL_TRUE, 0, sizeof(float2) * next_num * batch, temp_out, 0, NULL, &readBuf_event);
  checkError(status, "Failed to copy data from device");

  status = clFinish(queue1);
  checkError(status, "failed to finish reading buffer using PCIe");
  /* Calculating kernel execution times from cl_events */

  // Time to transfer signal from host to global memory
  // Also overlaps generation of W and filter transformation
  cl_ulong writeBuf_start = 0.0, writeBuf_end = 0.0;
  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &writeBuf_start, NULL);
  clGetEventProfilingInfo(writeBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &writeBuf_end, NULL);

  fft_time.pcie_write_t = (cl_double)(writeBuf_end - writeBuf_start) * (cl_double)(1e-06);

  printf("-- Timing:\n");
  printf("\tSignal Host to DDR: %lfms\n", fft_time.pcie_write_t);

  // Time to transform signal
  cl_ulong kernelSignal_start = 0.0f, kernelSignal_end = 0.0f;
  clGetEventProfilingInfo(startSignal_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelSignal_start, NULL);
  clGetEventProfilingInfo(stopSignal_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelSignal_end, NULL);

  fft_time.exec_t = (cl_double)(kernelSignal_end - kernelSignal_start) * (cl_double)(1e-06); 
  printf("\tKernel Execution: %lfms\n", fft_time.exec_t);

  // Time to transfer results from device to host 
  cl_ulong readBuf_start = 0.0, readBuf_end = 0.0;
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &readBuf_start, NULL);
  clGetEventProfilingInfo(readBuf_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &readBuf_end, NULL);

  fft_time.pcie_read_t = (cl_double)(readBuf_end - readBuf_start) * (cl_double)(1e-06);
  clReleaseEvent(writeBuf_event);

  printf("\tSignal DDR to Host: %lfms\n", fft_time.pcie_read_t);

  // Time to transform Filter + Signal
  // Time to generate W and Filter
  printf("\n-- Additional Timings:\n");
  cl_ulong kernelgenW_start = 0.0f, kernelgenW_end = 0.0f;
  clGetEventProfilingInfo(genW_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelgenW_start, NULL);
  clGetEventProfilingInfo(genW_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelgenW_end, NULL);

  double genW_t = (cl_double)(kernelgenW_end - kernelgenW_start) * (cl_double)(1e-06); 
  printf("\tKernel Execution: Gen W and Filter creation %lfms\n", genW_t);

  // Reorder output without the padded zeroes
  for(unsigned i = 0; i < batch; i++){
    for(unsigned j = 0; j < num; j++){
      out[(i * num) + j] = temp_out[(i * next_num) + j];
    }
  }

  if(d_bufW)
  	clReleaseMemObject(d_bufW);
  if(d_Filter)
  	clReleaseMemObject(d_Filter);
  if(d_Signal)
  	clReleaseMemObject(d_Signal);
  if(d_Out)
  	clReleaseMemObject(d_Out);

  if(kernel_gen_W)
    clReleaseKernel(kernel_gen_W);
  if(kernel_fetch)
    clReleaseKernel(kernel_fetch);
  if(kernel_mod)
    clReleaseKernel(kernel_mod);
  if(kernel_fft1da)
    clReleaseKernel(kernel_fft1da);
  if(kernel_mult)
    clReleaseKernel(kernel_mult);
  if(kernel_fftinv1d)
    clReleaseKernel(kernel_fftinv1d);
  if(kernel_scale)
    clReleaseKernel(kernel_scale);
  if(kernel_demod)
    clReleaseKernel(kernel_demod);

  queue_cleanup();
  free(temp_in);
  free(temp_out);

  fft_time.valid = 1;
  return fft_time;
}