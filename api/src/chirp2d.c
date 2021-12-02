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
 * \brief  compute an out-of-place single precision complex 2D-FFT on the FPGA
 * \param  N    : unsigned integer to the number of points in FFT1d  
 * \param  inp  : float2 pointer to input data of size N
 * \param  out  : float2 pointer to output data of size N
 * \param  inv  : toggle for backward transforms
 * \param  batch : number of batched executions of 2D FFT
 * \return fpga_t : time taken in milliseconds for data transfers and execution
 */
fpga_t fftfpgaf_c2c_chirp2d_bram(const unsigned num, const float2 *inp, float2 *out, const bool inv, const unsigned batch){

  fpga_t fft_time = {0.0, 0.0, 0.0, 0.0, 0.0, 0};

  if(inp == NULL || out == NULL || (num < 8))
    return fft_time;

  cl_int status = 0;
  unsigned next_num = next_power_of_two(num);
  unsigned chirp_num = next_second_power_of_two(num);

  printf("\tNumber of points: %ux%u\n\tNext Power of 2: %ux%u\n\tChirp Power of 2: %ux%u\n\tBatch: %u\n", num, num, next_num, next_num, chirp_num, chirp_num, batch);
  
  // Zero array to initialize device buffer for W
  float2 temp[chirp_num];
  for(unsigned i = 0; i < chirp_num; i++){
    temp[i].x = 0.0f;
    temp[i].y = 0.0f;
  }

  // Zero fill input, output buffers padded to nearest powers of 2
  float2 *temp_in = malloc(sizeof(float2) * next_num * next_num * batch);
  float2 *temp_out = malloc(sizeof(float2) * next_num * next_num * batch);
  for(unsigned i = 0; i < batch * next_num * next_num; i++){
    temp_in[i].x = 0.0f;
    temp_in[i].y = 0.0f;
    temp_out[i].x = 0.0f;
    temp_out[i].y = 0.0f;
  }

  // copy 31x31 buffer into a 32x32
  for(unsigned i = 0; i < batch; i++){
    for(unsigned j = 0; j < num; j++){   
      for(unsigned k = 0; k < num; k++){ 

        unsigned index_in = (i * num * num) + (j * num) + k;
        unsigned index_out = (i * next_num * next_num) + (j * next_num) + k;

        temp_in[index_out].x = inp[index_in].x;
        temp_in[index_out].y = inp[index_in].y;
      }
    }
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

  cl_mem d_Filter_fourier = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_CHANNEL_1_INTELFPGA, sizeof(float2) * chirp_num, NULL, &status);
  checkError(status, "Failed to allocate buffer for Filter in fourier\n");

  cl_mem d_Signal = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_CHANNEL_3_INTELFPGA, sizeof(float2) * next_num * next_num * batch, NULL, &status);
  checkError(status, "Failed to allocate Signal buffer\n");

  cl_mem d_Out = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_CHANNEL_4_INTELFPGA, sizeof(float2) * next_num * next_num * batch, NULL, &status);
  checkError(status, "Failed to allocate Out buffer\n");

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
  unsigned filter_count = 1;
  cl_kernel kernel_fetch = clCreateKernel(program, "fetch1", &status);
  checkError(status, "Failed to create fetch1 kernel");
  status = clSetKernelArg(kernel_fetch, 0, sizeof(cl_mem), (void *)&d_Filter);
  checkError(status, "Failed to set kernel fetch1 arg 0");
  status = clSetKernelArg(kernel_fetch, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel fetch1 arg 1"); 
  unsigned modulate = NO_MODULATE;
  status = clSetKernelArg(kernel_fetch, 2, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel fetch1 arg 2"); 
  status = clSetKernelArg(kernel_fetch, 3, sizeof(cl_uint), (void *)&filter_count);
  checkError(status, "Failed to set kernel fetch1 arg 3");

  cl_kernel kernel_mod1 = clCreateKernel(program, "modulate1", &status);
  checkError(status, "Failed to create modulate1 kernel");
  status = clSetKernelArg(kernel_mod1, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel modulate1 arg 0");
  status = clSetKernelArg(kernel_mod1, 1, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel modulate1 arg 1"); 
  status = clSetKernelArg(kernel_mod1, 2, sizeof(cl_uint), (void *)&filter_count);
  checkError(status, "Failed to set kernel modulate1 arg 2"); 

  unsigned fft1d_1_count = filter_count;
  unsigned isFilter = 1;
  int inverse = 0;
  cl_kernel kernel_fft1d_1 = clCreateKernel(program, "fft1d_1", &status);
  checkError(status, "Failed to create fft1d_1 kernel");
  status = clSetKernelArg(kernel_fft1d_1, 0, sizeof(cl_mem), (void *)&d_Filter_fourier);
  checkError(status, "Failed to set kernel fft1d_1 arg 0"); 
  status = clSetKernelArg(kernel_fft1d_1, 1, sizeof(cl_uint),(void *)&isFilter);
  checkError(status, "Failed to set kernel fft1d_1 arg 1"); 
  status = clSetKernelArg(kernel_fft1d_1, 2, sizeof(cl_uint), (void *)&fft1d_1_count);
  checkError(status, "Failed to set kernel fft1d_1 arg 2"); 
  status = clSetKernelArg(kernel_fft1d_1, 3, sizeof(cl_int), (void *)&inverse);
  checkError(status, "Failed to set kernel fft1d_1 arg 3"); 

  unsigned total_mult1_count = batch * next_num;
  // Filter is stored in BRAM, then multiplied with count * signals 
  cl_kernel kernel_mult1 = clCreateKernel(program, "multiplication1", &status);
  checkError(status, "Failed to create multiplication1 kernel");
  status = clSetKernelArg(kernel_mult1, 0, sizeof(cl_mem), (void *)&d_Filter_fourier);
  checkError(status, "Failed to set kernel mult1 arg 0"); 
  status = clSetKernelArg(kernel_mult1, 1, sizeof(cl_uint), (void *)&total_mult1_count);
  checkError(status, "Failed to set kernel mult1 arg 1"); 
 
  // Arguments for Signal Transformation
  int inverse_sig = 1;
  unsigned ifft1d_1_count = batch * next_num;
  cl_kernel kernel_ifft1d_1 = clCreateKernel(program, "ifft1d_1", &status);
  checkError(status, "Failed to create ifft1d_1 kernel");
  status = clSetKernelArg(kernel_ifft1d_1, 0, sizeof(cl_uint), (void *)&ifft1d_1_count);
  checkError(status, "Failed to set kernel ifft1d_1 arg 0"); 
  status = clSetKernelArg(kernel_ifft1d_1, 1, sizeof(cl_int), (void *)&inverse_sig);
  checkError(status, "Failed to set kernel ifft1d_1 arg 1");

  float factor = 1.0f / (float)chirp_num;

  unsigned scale1_count = batch * next_num;
  cl_kernel kernel_scale1 = clCreateKernel(program, "scale1", &status);
  status = clSetKernelArg(kernel_scale1, 0, sizeof(cl_float), (void *)&factor);
  checkError(status, "Failed to set kernel scale1 arg 0"); 
  status = clSetKernelArg(kernel_scale1, 1, sizeof(cl_uint), (void *)&scale1_count);
  checkError(status, "Failed to set kernel scale1 arg 1"); 

  unsigned demod1_count = batch * next_num;
  cl_kernel kernel_demod1 = clCreateKernel(program, "demodulate1", &status);
  status = clSetKernelArg(kernel_demod1, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel demodulate1 arg 0");
  status = clSetKernelArg(kernel_demod1, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel demodulate1 arg 1"); 
  status = clSetKernelArg(kernel_demod1, 2, sizeof(cl_uint), (void *)&demod1_count);
  checkError(status, "Failed to set kernel demodulate1 arg 2"); 

  cl_kernel kernel_transpose = clCreateKernel(program, "transpose", &status);
  status = clSetKernelArg(kernel_transpose, 0, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel transpose arg 0"); 
  status = clSetKernelArg(kernel_transpose, 1, sizeof(cl_uint), (void *)&batch);
  checkError(status, "Failed to set kernel transpose arg 1"); 

  unsigned mod2_count = batch * next_num;
  cl_kernel kernel_mod2 = clCreateKernel(program, "modulate2", &status);
  checkError(status, "Failed to create modulate2 kernel");
  status = clSetKernelArg(kernel_mod2, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel modulate2 arg 0");
  status = clSetKernelArg(kernel_mod2, 1, sizeof(cl_uint), (void *)&mod2_count);
  checkError(status, "Failed to set kernel modulate2 arg 1"); 

  unsigned fft1d_2_count = (batch * next_num);
  cl_kernel kernel_fft1d_2 = clCreateKernel(program, "fft1d_2", &status);
  checkError(status, "Failed to create fft1d_2 kernel");
  status = clSetKernelArg(kernel_fft1d_2, 0, sizeof(cl_uint), (void *)&fft1d_2_count);
  checkError(status, "Failed to set kernel fft1d_2 arg 0"); 
  inverse = 0;
  status = clSetKernelArg(kernel_fft1d_2, 1, sizeof(cl_int), (void *)&inverse);
  checkError(status, "Failed to set kernel fft1d_2 arg 1"); 

  unsigned mult2_count = batch * next_num;
  // Filter is stored in BRAM, then multiplied with count * signals 
  cl_kernel kernel_mult2 = clCreateKernel(program, "multiplication2", &status);
  checkError(status, "Failed to create multiplication2 kernel");
  status = clSetKernelArg(kernel_mult2, 0, sizeof(cl_mem), (void *)&d_Filter_fourier);
  checkError(status, "Failed to set kernel mult2 arg 0"); 
  status = clSetKernelArg(kernel_mult2, 1, sizeof(cl_uint), (void *)&mult2_count);
  checkError(status, "Failed to set kernel mult2 arg 1"); 

  inverse_sig = 1;
  unsigned ifft1d_2_count = batch * next_num;
  cl_kernel kernel_ifft1d_2 = clCreateKernel(program, "ifft1d_2", &status);
  checkError(status, "Failed to create ifft1d_2 kernel");
  status = clSetKernelArg(kernel_ifft1d_2, 0, sizeof(cl_uint), (void *)&ifft1d_2_count);
  checkError(status, "Failed to set kernel ifft1d_2 arg 0"); 
  status = clSetKernelArg(kernel_ifft1d_2, 1, sizeof(cl_int), (void *)&inverse_sig);
  checkError(status, "Failed to set kernel ifft1d_2 arg 1");

  unsigned scale2_count = batch * next_num;
  cl_kernel kernel_scale2 = clCreateKernel(program, "scale2", &status);
  status = clSetKernelArg(kernel_scale2, 0, sizeof(cl_float), (void *)&factor);
  checkError(status, "Failed to set kernel scale1 arg 0"); 
  status = clSetKernelArg(kernel_scale2, 1, sizeof(cl_uint), (void *)&scale2_count);
  checkError(status, "Failed to set kernel scale1 arg 1"); 

  unsigned demod2_count = batch * next_num;
  cl_kernel kernel_demod2 = clCreateKernel(program, "demodulate2", &status);
  status = clSetKernelArg(kernel_demod2, 0, sizeof(cl_mem), (void *)&d_bufW);
  checkError(status, "Failed to set kernel demodulate2 arg 0");
  status = clSetKernelArg(kernel_demod2, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel demodulate2 arg 1"); 
  status = clSetKernelArg(kernel_demod2, 2, sizeof(cl_uint), (void *)&demod2_count);
  checkError(status, "Failed to set kernel demodulate2 arg 2"); 

  unsigned tranStore_count = batch;
  cl_kernel kernel_tranStore = clCreateKernel(program, "transposeStore", &status);
  status = clSetKernelArg(kernel_tranStore, 0, sizeof(cl_mem), (void *)&d_Out);
  checkError(status, "Failed to set kernel transpose Store arg 0");
  status = clSetKernelArg(kernel_tranStore, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel transpose Store arg 1"); 
  status = clSetKernelArg(kernel_tranStore, 2, sizeof(cl_uint), (void *)&tranStore_count);
  checkError(status, "Failed to set kernel transpose Store arg 2"); 

  // Overlap transferring signal with W generation and Filter transformation
  cl_event startFilter_event, stopFilter_event;
  cl_event startSignal_event, stopSignal_event;
  cl_event writeBuf_event, genW_event;
  status = clEnqueueWriteBuffer(queue1, d_Signal, CL_FALSE, 0, sizeof(float2) * next_num * next_num * batch, temp_in, 0, NULL, &writeBuf_event);
  checkError(status, "failed to finish writing inp data to device buffer d_Signal");

  status = clEnqueueTask(queue2, kernel_gen_W, 0, NULL, &genW_event);
  checkError(status, "Failed to launch gen_W kernel");
  status = clFinish(queue2);
  checkError(status, "Failed to finish queue2");
  /*
  *  Transform Filter and store in bitreversed order in multiplication kernel
  */
  // kernel_mult and kernel_fft1da continues to execute also the signal
  status = clEnqueueTask(queue4, kernel_fft1d_1, 0, NULL, &stopFilter_event);
  checkError(status, "Failed to launch fft1d kernel");
  status = clEnqueueTask(queue3, kernel_mod1, 0, NULL, NULL);
  checkError(status, "Failed to launch modulate kernel");
  status = clEnqueueTask(queue2, kernel_fetch, 0, NULL, &startFilter_event);
  checkError(status, "Failed to launch fetch kernel");

  // fft1d and mult continue to process the signal data also
  status = clFinish(queue4);
  checkError(status, "Failed to finish queue4");
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
  modulate = MODULATE;
  unsigned fetch_count = batch * next_num;
  status = clSetKernelArg(kernel_fetch, 0, sizeof(cl_mem), (void *)&d_Signal);
  checkError(status, "Failed to set kernel fetch arg 0");
  status = clSetKernelArg(kernel_fetch, 1, sizeof(cl_uint), (void *)&num);
  checkError(status, "Failed to set kernel fetch arg 1"); 
  status = clSetKernelArg(kernel_fetch, 2, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel fetch arg 2"); 
  status = clSetKernelArg(kernel_fetch, 3, sizeof(cl_uint), (void *)&fetch_count);
  checkError(status, "Failed to set kernel fetch arg 3");

  unsigned kernel_mod1_count = batch * next_num;
  status = clSetKernelArg(kernel_mod1, 1, sizeof(cl_uint), (void *)&modulate);
  checkError(status, "Failed to set kernel modulate1 arg 1"); 
  status = clSetKernelArg(kernel_mod1, 2, sizeof(cl_uint), (void *)&kernel_mod1_count);
  checkError(status, "Failed to set kernel modulate1 arg 2"); 

  fft1d_1_count = batch * next_num;
  isFilter = 0;
  inverse = 0;
  kernel_fft1d_1 = clCreateKernel(program, "fft1d_1", &status);
  checkError(status, "Failed to create fft1d_1 kernel");
  status = clSetKernelArg(kernel_fft1d_1, 0, sizeof(cl_mem), (void *)&d_Filter_fourier);
  checkError(status, "Failed to set kernel fft1d_1 arg 0"); 
  status = clSetKernelArg(kernel_fft1d_1, 1, sizeof(cl_uint),(void *)&isFilter);
  checkError(status, "Failed to set kernel fft1d_1 arg 1"); 
  status = clSetKernelArg(kernel_fft1d_1, 2, sizeof(cl_uint), (void *)&fft1d_1_count);
  checkError(status, "Failed to set kernel fft1d_1 arg 2"); 
  status = clSetKernelArg(kernel_fft1d_1, 3, sizeof(cl_int), (void *)&inverse);
  checkError(status, "Failed to set kernel fft1d_1 arg 3");

  // Queues 4,5,7 are occupied by FFT1da, mult and demod kernels respectively
  status = clEnqueueTask(queue15, kernel_tranStore, 0, NULL, &stopSignal_event);
  checkError(status, "Failed to launch tranStore kernel");
  status = clEnqueueTask(queue14, kernel_demod2, 0, NULL, NULL);
  checkError(status, "Failed to launch demod2 kernel");
  status = clEnqueueTask(queue13, kernel_scale2, 0, NULL, NULL);
  checkError(status, "Failed to launch scale2 kernel");
  status = clEnqueueTask(queue12, kernel_ifft1d_2, 0, NULL, NULL);
  checkError(status, "Failed to launch ifft1d_2 kernel");
  status = clEnqueueTask(queue11, kernel_mult2, 0, NULL, NULL);
  checkError(status, "Failed to launch multiplication 2 kernel");
  status = clEnqueueTask(queue10, kernel_fft1d_2, 0, NULL, NULL);
  checkError(status, "Failed to launch fft1d_2 kernel");
  status = clEnqueueTask(queue9, kernel_mod2, 0, NULL, NULL);
  checkError(status, "Failed to launch mod2 kernel");
  status = clEnqueueTask(queue8, kernel_transpose, 0, NULL, NULL);
  checkError(status, "Failed to launch transpose kernel");
  status = clEnqueueTask(queue7, kernel_demod1, 0, NULL, NULL);
  checkError(status, "Failed to launch demod kernel");
  status = clEnqueueTask(queue6, kernel_scale1, 0, NULL, NULL);
  checkError(status, "Failed to launch scale1 kernel");
  status = clEnqueueTask(queue5, kernel_ifft1d_1, 0, NULL, NULL);
  checkError(status, "Failed to launch ifft1d_1 kernel");
  status = clEnqueueTask(queue4, kernel_mult1, 0, NULL, NULL);
  checkError(status, "Failed to launch multiplication kernel");
  status = clEnqueueTask(queue3, kernel_fft1d_1, 0, NULL, NULL);
  checkError(status, "Failed to launch fft1d_1 kernel");
  status = clEnqueueTask(queue2, kernel_mod1, 0, NULL, NULL);
  checkError(status, "Failed to launch modulate kernel");
  status = clEnqueueTask(queue1, kernel_fetch, 0, NULL, &startSignal_event);
  checkError(status, "Failed to launch ifft kernel");
  
  status = clFinish(queue14);
  checkError(status, "Failed to finish queue14");
  status = clFinish(queue13);
  checkError(status, "Failed to finish queue13");
  status = clFinish(queue12);
  checkError(status, "Failed to finish queue12");
  status = clFinish(queue11);
  checkError(status, "Failed to finish queue11");
  status = clFinish(queue10);
  checkError(status, "Failed to finish queue10");
  status = clFinish(queue9);
  checkError(status, "Failed to finish queue9");
  status = clFinish(queue8);
  checkError(status, "Failed to finish queue8");
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
  status = clEnqueueReadBuffer(queue1, d_Out, CL_TRUE, 0, sizeof(float2) * next_num * next_num * batch, temp_out, 0, NULL, &readBuf_event);
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
  printf("\tGen W and Filter creation: %lfms\n", genW_t);

  cl_ulong kernelFilter_start = 0.0f, kernelFilter_end = 0.0f;
  clGetEventProfilingInfo(startFilter_event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &kernelFilter_start, NULL);
  clGetEventProfilingInfo(stopFilter_event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &kernelFilter_end, NULL);

  double filter_fourier_t = (cl_double)(kernelFilter_end - kernelFilter_start) * (cl_double)(1e-06); 
  printf("\tFilter transformation: %lfms\n\n", filter_fourier_t);

  // Reorder output without the padded zeroes
  for(unsigned i = 0; i < batch; i++){
    for(unsigned j = 0; j < num; j++){
      for(unsigned k = 0; k < num; k++){
        out[(i * num * num) + (j * num) + k].x = temp_out[(i * next_num * next_num) + (j * next_num) + k].x;
        out[(i * num * num) + (j * num) + k].y = temp_out[(i * next_num * next_num) + (j * next_num) + k].y;
      }
    }
  }

  if(d_bufW)
  	clReleaseMemObject(d_bufW);
  if(d_Filter)
  	clReleaseMemObject(d_Filter);
  if(d_Filter_fourier)
  	clReleaseMemObject(d_Filter_fourier);
  if(d_Signal)
  	clReleaseMemObject(d_Signal);
  if(d_Out)
  	clReleaseMemObject(d_Out);

  if(kernel_gen_W)
    clReleaseKernel(kernel_gen_W);
  if(kernel_fetch)
    clReleaseKernel(kernel_fetch);
  if(kernel_mod1)
    clReleaseKernel(kernel_mod1);
  if(kernel_fft1d_1)
    clReleaseKernel(kernel_fft1d_1);
  if(kernel_mult1)
    clReleaseKernel(kernel_mult1);
  if(kernel_ifft1d_1)
    clReleaseKernel(kernel_ifft1d_1);
  if(kernel_scale1)
    clReleaseKernel(kernel_scale1);
  if(kernel_demod1)
    clReleaseKernel(kernel_demod1);
  if(kernel_transpose)
    clReleaseKernel(kernel_transpose);
  if(kernel_mod2)
    clReleaseKernel(kernel_mod2);
  if(kernel_fft1d_2)
    clReleaseKernel(kernel_fft1d_2);
  if(kernel_mult2)
    clReleaseKernel(kernel_mult2);
  if(kernel_ifft1d_2)
    clReleaseKernel(kernel_ifft1d_2);
  if(kernel_scale2)
    clReleaseKernel(kernel_scale2);
  if(kernel_demod2)
    clReleaseKernel(kernel_demod2);
  if(kernel_tranStore)
    clReleaseKernel(kernel_tranStore);

  queue_cleanup();
  free(temp_in);
  free(temp_out);

  fft_time.valid = 1;
  return fft_time;
}