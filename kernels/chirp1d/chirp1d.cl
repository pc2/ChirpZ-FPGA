// Author: Arjun Ramaswami
#include "fft_config.h"
#include "../common/fft_8.cl" 
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel float2 chaninmod[POINTS]; 
channel float2 chaninfft1da[POINTS]; 
channel float2 chaninmult[POINTS]; 
channel float2 chaninfftinv1da[POINTS]; 
channel float2 chaninscale[POINTS]; 
channel float2 chanindemod[POINTS]; 

/*
 * \brief Bitreversal of a number, obtained as output from FFT1D Kernel
 * \ret   number obtained on bitreversal
 */
unsigned bit_reversed(unsigned x, unsigned bits) {
  unsigned y = 0;
  #pragma unroll 
  for (uint i = 0; i < bits; i++) {
    y <<= 1;
    y |= x & 1;
    x >>= 1;
  }
  y &= ((1 << bits) - 1);
  return y;
}

/*
 * \brief Reverse the 8 point complex array
 * \ret   8 point complex array that is reversed
 */
float2x8 reverse(float2x8 data){ // 8-point complex array
  float2x8 test;

  test.i7 = data.i0;
  test.i6 = data.i1;
  test.i5 = data.i2;
  test.i4 = data.i3;
  test.i3 = data.i4;
  test.i2 = data.i5;
  test.i1 = data.i6;
  test.i0 = data.i7;

  data.i0 = test.i0;
  data.i1 = test.i1;
  data.i2 = test.i2;
  data.i3 = test.i3;
  data.i4 = test.i4;
  data.i5 = test.i5;
  data.i6 = test.i6;
  data.i7 = test.i7;

  return data;
}

/*
 * \brief Generate W and Filter arrays
 */
kernel void gen_W_Filter(
  __global volatile float2 * restrict W,       // W[NEAREST_POW_OF_2] in DDR
  __global volatile float2 * restrict filter,  // filter[2 * NEAREST_POW_OF_2] DDR
  unsigned int num,                           // prime or non-prime number
  unsigned int inv){                          // inverse transform  

  // Half of power of two to create mirror image for the convolution filter 
  float2 chirp_filter[NEAREST_POW_OF_2];        
  float2 chirp_filter_rev[NEAREST_POW_OF_2];

  float sign = (inv == 1) ? 1.0f : -1.0f;

  for(unsigned i = 0; i < (NEAREST_POW_OF_2 / 8); i++){

    /*
     * Create W in DDR and the first half of the filter in BRAM in a loop
     */
    #pragma unroll 8
    for(unsigned j = 0; j < 8; j++){
      unsigned index = (i * 8) + j;
      float2 temp[8];
      temp[j].x = (index < num) ? cos(M_PI_F * index * index / num) : 0.0f;
      temp[j].y = (index < num) ? sign * sin(M_PI_F * index * index / num) : 0.0f;

      chirp_filter[index].x = temp[j].x;
      chirp_filter[index].y = (index < num) ? (-1.0f * temp[j].y) : 0.0f;

      W[index] = temp[j];
    }
    
    float2x8 data, data_rev;
    data.i0 = chirp_filter[(i * 8) + 0];
    data.i1 = chirp_filter[(i * 8) + 1];
    data.i2 = chirp_filter[(i * 8) + 2];
    data.i3 = chirp_filter[(i * 8) + 3];
    data.i4 = chirp_filter[(i * 8) + 4];
    data.i5 = chirp_filter[(i * 8) + 5];
    data.i6 = chirp_filter[(i * 8) + 6];
    data.i7 = chirp_filter[(i * 8) + 7];

    // Inverting the first half of the filter to create the mirror image
    // Inversion is along the zeroth frequency
    // - requires shifting by 1, hence the index_rev = () + 1
    // - also the border check to wrap the final element to not cause segfault
    // - therefore, the explicit setting first filter point to 0 after the loop 
    data_rev = reverse(data);

    unsigned index_rev = ((NEAREST_POW_OF_2) - ((i+1) * 8)) + 1;
    chirp_filter_rev[(index_rev + 0) & (NEAREST_POW_OF_2 - 1)] = data_rev.i0;
    chirp_filter_rev[(index_rev + 1) & (NEAREST_POW_OF_2 - 1)] = data_rev.i1;
    chirp_filter_rev[(index_rev + 2) & (NEAREST_POW_OF_2 - 1)] = data_rev.i2;
    chirp_filter_rev[(index_rev + 3) & (NEAREST_POW_OF_2 - 1)] = data_rev.i3;
    chirp_filter_rev[(index_rev + 4) & (NEAREST_POW_OF_2 - 1)] = data_rev.i4;
    chirp_filter_rev[(index_rev + 5) & (NEAREST_POW_OF_2 - 1)] = data_rev.i5;
    chirp_filter_rev[(index_rev + 6) & (NEAREST_POW_OF_2 - 1)] = data_rev.i6;
    chirp_filter_rev[(index_rev + 7) & (NEAREST_POW_OF_2 - 1)] = data_rev.i7;
  }
  chirp_filter_rev[0].x = 0.0f;
  chirp_filter_rev[0].y = 0.0f;

  /*
  * Storing the convolution filter by each half to DDR
  * - could also write to one buffer and store to ddr for increased burst
  * - tradeoff here to reduce logic / mem since the array is smaller than FFT3D
  */
  #pragma unroll 8
  for(unsigned i = 0; i < NEAREST_POW_OF_2; i++){
    float2 temp;
    temp.x = chirp_filter[i].x;
    temp.y = chirp_filter[i].y;

    filter[i] = temp;
  }

  #pragma unroll 8
  for(unsigned i = 0; i < NEAREST_POW_OF_2; i++){
    float2 temp;
    temp.x = chirp_filter_rev[i].x;
    temp.y = chirp_filter_rev[i].y;

    filter[NEAREST_POW_OF_2 + i] = temp;
  }
}

/*
 * \brief Kernel that fetches data from global memory 
 * \param src: pointer to buffer in global memory for filter or signal
 * \param num: the sample size of the signal
 * \param modulate: set to 1 for modulating signal with W array, here just to set the values above num to 0.0 
 */
kernel void fetch(
  global volatile float2 * restrict src, // pointer to signal or filter in DDR
  unsigned int num,                      // number of sample signal points
  unsigned int modulate,                 // set to 1 when modulating signal
  unsigned int count){                   // number of iterations

  unsigned next_num = NEAREST_POW_OF_2;
  unsigned chirp_num = NEAREST_POW_OF_2 + NEAREST_POW_OF_2;

  #pragma loop_coalesce
  for(unsigned row = 0; row < count; row++){
    for(unsigned step = 0; step < (chirp_num / 8); step++){

      float2 data[8];
      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){
        // signal is only NEAREST_POW_2 long, hence the condition sets to 0.0 
        // filter is 2* NEAREST_POW_2 long
        float2 tempzero;
        tempzero.x = 0.0f;
        tempzero.y = 0.0f;

        // every chirp_num points have to be padded with zeroes
        data[i] = ( ((step * 8) + i) >= next_num && modulate) ? tempzero: src[(row * NEAREST_POW_OF_2) + (step * 8) + i];
      }

      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){
        write_channel_intel(chaninmod[i], data[i]);
      }
    }
  }
}

/*
 * \brief Modulate signal or passthrough filter to the FFT kernel
 * \param W: pointer to W generated
 * \param modulate: set to 1 if modulating signal by W
 * \ret In bitreversed order
 */
kernel void modulate(
  global volatile float2 * restrict W, // pointer to W array in DDR
  unsigned int modulate,               // set to 1 when modulating signal
  unsigned int count){                 // number of iterations

  unsigned next_num = NEAREST_POW_OF_2;
  unsigned chirp_num = NEAREST_POW_OF_2 + NEAREST_POW_OF_2;

  float2 bufW[NEAREST_POW_OF_2 + NEAREST_POW_OF_2]; // W

  // When not modulating signal, fetch W array from DDR and store in BRAM
  // Implementing this within the nested loops with an else stm, increases loop latency 5x - 6x fold
  #pragma unroll 8
  for(unsigned step = 0; step < chirp_num; step++){

    float2 tempzero;
    tempzero.x = 0.0f;
    tempzero.y = 0.0f;

    bufW[step] = (step < next_num) ? W[step] : tempzero;
  }

  for(unsigned row = 0; row < count; row++){

    float2 buf[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];  // Signal or Filter
    for(unsigned step = 0; step < (chirp_num / 8); step++){
    
      float2 data[8];

      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){
        data[i] = read_channel_intel(chaninmod[i]);
      }

      // Modulate signal with W otherwise passthrough (for filter)
      if(modulate == 1){
        #pragma unroll 8
        for(unsigned i = 0; i < 8; i++){
          unsigned index = step * 8;
          float x = bufW[index + i].x;
          float y = bufW[index + i].y;
          float a = data[i].x; 
          float b = data[i].y; 

          data[i].x = (x * a) - (y * b);
          data[i].y = (x * b) + (y * a);
        }
      }
      
      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){
        buf[((step * 8) + i) & (chirp_num - 1)].x = data[i].x;
        buf[((step * 8) + i) & (chirp_num - 1)].y = data[i].y;
      }
    }
    
    // bitreverse as input to FFT1D
    for(unsigned step = 0; step < (chirp_num / 8); step++){
      write_channel_intel(chaninfft1da[0], buf[step]);               // 0
      write_channel_intel(chaninfft1da[1], buf[4 * chirp_num / 8 + step]); // 32
      write_channel_intel(chaninfft1da[2], buf[2 * chirp_num / 8 + step]); // 16
      write_channel_intel(chaninfft1da[3], buf[6 * chirp_num / 8 + step]); // 48
      write_channel_intel(chaninfft1da[4], buf[chirp_num / 8 + step]); // 8
      write_channel_intel(chaninfft1da[5], buf[5 * chirp_num / 8 + step]); // 40
      write_channel_intel(chaninfft1da[6], buf[3 * chirp_num / 8 + step]); // 24
      write_channel_intel(chaninfft1da[7], buf[7 * chirp_num / 8 + step]); // 54
    }
  }   // rows
}

kernel void fft1da(
  unsigned int count,       // number of FFTs
  unsigned int inverse) {   // set to 1 for iFFT

  /* The FFT engine requires a sliding window array for data reordering; data 
   * stored in this array is carried across loop iterations and shifted by one 
   * element every iteration; all loop dependencies derived from the uses of 
   * this array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + 8 * (LOGN - 2)];

  /* This is the main loop. It runs 'count' back-to-back FFT transforms
   * In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
   * additional iterations to drain the last outputs 
   * (see comments attached to the FFT engine)
   *
   * The compiler leverages pipeline parallelism by overlapping the 
   * iterations of this loop - launching one iteration every clock cycle
   */

  for (unsigned i = 0; i < count * (N / 8) + N / 8 - 1; i++) {

    /* As required by the FFT engine, gather input data from 8 distinct 
     * segments of the input buffer; for simplicity, this implementation 
     * does not attempt to coalesce memory accesses and this leads to 
     * higher resource utilization (see the fft2d example for advanced 
     * memory access techniques)
     */

    int base = (i / (N / 8)) * N;
    int offset = i % (N / 8);

    float2x8 data;
    // Perform memory transfers only when reading data in range
    if (i < count * (N / 8)) {
      data.i0 = read_channel_intel(chaninfft1da[0]);
      data.i1 = read_channel_intel(chaninfft1da[1]);
      data.i2 = read_channel_intel(chaninfft1da[2]);
      data.i3 = read_channel_intel(chaninfft1da[3]);
      data.i4 = read_channel_intel(chaninfft1da[4]);
      data.i5 = read_channel_intel(chaninfft1da[5]);
      data.i6 = read_channel_intel(chaninfft1da[6]);
      data.i7 = read_channel_intel(chaninfft1da[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Perform one step of the FFT engine
    data = fft_step(data, i % (N / 8), fft_delay_elements, inverse, LOGN); 

    /* Store data back to memory. FFT engine outputs are delayed by 
     * N / 8 - 1 steps, hence gate writes accordingly
     */

    if (i >= N / 8 - 1) {
      write_channel_intel(chaninmult[0], data.i0);
      write_channel_intel(chaninmult[1], data.i1);
      write_channel_intel(chaninmult[2], data.i2);
      write_channel_intel(chaninmult[3], data.i3);
      write_channel_intel(chaninmult[4], data.i4);
      write_channel_intel(chaninmult[5], data.i5);
      write_channel_intel(chaninmult[6], data.i6);
      write_channel_intel(chaninmult[7], data.i7);
    }
  }
}

/*
 * \brief point-wise multiplication of signal with filter
 *        the filter is stored in bit-reversed order 
 *        hence the signal is multiplied before stored back in normal order
 */
kernel void multiplication(
  unsigned int count){ // number of iterations

  unsigned chirp_num = NEAREST_POW_OF_2 + NEAREST_POW_OF_2;
  float2 filter_bitrev[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];

  // Store Filter in BRAM
  for(unsigned step = 0; step < (chirp_num / 8); step++){
    #pragma unroll 8
    for(unsigned i = 0; i < 8; i++){
      filter_bitrev[(step * 8) + i] = read_channel_intel(chaninmult[i]);
    }
  }

  for(unsigned row = 0; row < count; row++){

    float2 tempBuf1[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];
    float2 tempBuf2[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];
    // Point-wise multiplication of signal and filter
    for(unsigned step = 0; step < (chirp_num / 8); step++){

      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){

        float2 temp = read_channel_intel(chaninmult[i]);
        float x = filter_bitrev[(step * 8) + i].x;
        float y = filter_bitrev[(step * 8) + i].y;

        float2 out;
        out.x = (x * temp.x) - (y * temp.y);
        out.y = (x * temp.y) + (y * temp.x);

        tempBuf1[(step * 8) + i].x = out.x;
        tempBuf1[(step * 8) + i].y = out.y;
      }
    }

    // Store signal back in normal order
    for(unsigned step = 0; step < chirp_num / 8; step++){

      unsigned i = step & (chirp_num / 8 - 1);

      unsigned index_out = i * 8;
      unsigned index0 = bit_reversed(index_out + 0, LOGN);
      unsigned index1 = bit_reversed(index_out + 1, LOGN);
      unsigned index2 = bit_reversed(index_out + 2, LOGN);
      unsigned index3 = bit_reversed(index_out + 3, LOGN);
      unsigned index4 = bit_reversed(index_out + 4, LOGN);
      unsigned index5 = bit_reversed(index_out + 5, LOGN);
      unsigned index6 = bit_reversed(index_out + 6, LOGN);
      unsigned index7 = bit_reversed(index_out + 7, LOGN);

      tempBuf2[(i * 8)]     = tempBuf1[index0];
      tempBuf2[(i * 8) + 1] = tempBuf1[index1];
      tempBuf2[(i * 8) + 2] = tempBuf1[index2];
      tempBuf2[(i * 8) + 3] = tempBuf1[index3];
      tempBuf2[(i * 8) + 4] = tempBuf1[index4];
      tempBuf2[(i * 8) + 5] = tempBuf1[index5];
      tempBuf2[(i * 8) + 6] = tempBuf1[index6];
      tempBuf2[(i * 8) + 7] = tempBuf1[index7];
    }

    // Output to FFT1D kernel in bitreverse order
    for(unsigned step = 0; step < (chirp_num / 8); step++){
      write_channel_intel(chaninfftinv1da[0], tempBuf2[step]);              // 0
      write_channel_intel(chaninfftinv1da[1], tempBuf2[4 * chirp_num / 8 + step]); // 32
      write_channel_intel(chaninfftinv1da[2], tempBuf2[2 * chirp_num / 8 + step]); // 16
      write_channel_intel(chaninfftinv1da[3], tempBuf2[6 * chirp_num / 8 + step]); // 48
      write_channel_intel(chaninfftinv1da[4], tempBuf2[chirp_num / 8 + step]); // 8
      write_channel_intel(chaninfftinv1da[5], tempBuf2[5 * chirp_num / 8 + step]); // 40
      write_channel_intel(chaninfftinv1da[6], tempBuf2[3 * chirp_num / 8 + step]); // 24
      write_channel_intel(chaninfftinv1da[7], tempBuf2[7 * chirp_num / 8 + step]); // 54
    }
  } // row
}

kernel void fft1db(
  unsigned count,  // number of FFTs
  int inverse) {   // set to 1 for iFFT

  /* The FFT engine requires a sliding window array for data reordering; data 
   * stored in this array is carried across loop iterations and shifted by one 
   * element every iteration; all loop dependencies derived from the uses of 
   * this array are simple transfers between adjacent array elements
   */

  float2 fft_delay_elements[N + 8 * (LOGN - 2)];

  /* This is the main loop. It runs 'count' back-to-back FFT transforms
   * In addition to the 'count * (N / 8)' iterations, it runs 'N / 8 - 1'
   * additional iterations to drain the last outputs 
   * (see comments attached to the FFT engine)
   *
   * The compiler leverages pipeline parallelism by overlapping the 
   * iterations of this loop - launching one iteration every clock cycle
   */

  for (unsigned i = 0; i < count * (N / 8) + N / 8 - 1; i++) {

    /* As required by the FFT engine, gather input data from 8 distinct 
     * segments of the input buffer; for simplicity, this implementation 
     * does not attempt to coalesce memory accesses and this leads to 
     * higher resource utilization (see the fft2d example for advanced 
     * memory access techniques)
     */

    int base = (i / (N / 8)) * N;
    int offset = i % (N / 8);

    float2x8 data;
    // Perform memory transfers only when reading data in range
    if (i < count * (N / 8)) {
      data.i0 = read_channel_intel(chaninfftinv1da[0]);
      data.i1 = read_channel_intel(chaninfftinv1da[1]);
      data.i2 = read_channel_intel(chaninfftinv1da[2]);
      data.i3 = read_channel_intel(chaninfftinv1da[3]);
      data.i4 = read_channel_intel(chaninfftinv1da[4]);
      data.i5 = read_channel_intel(chaninfftinv1da[5]);
      data.i6 = read_channel_intel(chaninfftinv1da[6]);
      data.i7 = read_channel_intel(chaninfftinv1da[7]);
    } else {
      data.i0 = data.i1 = data.i2 = data.i3 = 
                data.i4 = data.i5 = data.i6 = data.i7 = 0;
    }

    // Perform one step of the FFT engine
    data = fft_step(data, i % (N / 8), fft_delay_elements, inverse, LOGN); 

    /* Store data back to memory. FFT engine outputs are delayed by 
     * N / 8 - 1 steps, hence gate writes accordingly
     */

    if (i >= N / 8 - 1) {
      write_channel_intel(chaninscale[0], data.i0);
      write_channel_intel(chaninscale[1], data.i1);
      write_channel_intel(chaninscale[2], data.i2);
      write_channel_intel(chaninscale[3], data.i3);
      write_channel_intel(chaninscale[4], data.i4);
      write_channel_intel(chaninscale[5], data.i5);
      write_channel_intel(chaninscale[6], data.i6);
      write_channel_intel(chaninscale[7], data.i7);
    }
  }
}

/*
 *  \brief Scale output of FFT by signal sample size
 */
kernel void scale(
  float factor,     // factor to scale by (1 / num)
  unsigned count){  // number of iterations

  unsigned chirp_num = NEAREST_POW_OF_2 + NEAREST_POW_OF_2;
  
  for(unsigned row = 0; row < count; row++){

    float2 buf[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];
    // store result of ifft1d in buffer, which is in bitreversed order
    // therefore, needs chirp_num points and not NEAREST_POW_OF_2
    for(unsigned step = 0; step < chirp_num / 8; step++){

      unsigned row = step & (chirp_num / 8 - 1);

      #pragma unroll 8
      for(unsigned i = 0; i < 8; i++){
        float2 temp = read_channel_intel(chaninscale[i]);
        
        buf[(row * 8) + i].x = temp.x * factor;
        buf[(row * 8) + i].y = temp.y * factor;
      }
    }

    // Bitreverse to normal order and channel for demodulation
    for(unsigned step = 0; step < NEAREST_POW_OF_2 / 8; step++){

      unsigned i = step & (NEAREST_POW_OF_2 / 8 - 1);

      unsigned index_out = i * 8;
      unsigned index0 = bit_reversed(index_out + 0, LOGN);
      unsigned index1 = bit_reversed(index_out + 1, LOGN);
      unsigned index2 = bit_reversed(index_out + 2, LOGN);
      unsigned index3 = bit_reversed(index_out + 3, LOGN);
      unsigned index4 = bit_reversed(index_out + 4, LOGN);
      unsigned index5 = bit_reversed(index_out + 5, LOGN);
      unsigned index6 = bit_reversed(index_out + 6, LOGN);
      unsigned index7 = bit_reversed(index_out + 7, LOGN);

      write_channel_intel(chanindemod[0], buf[index0]);
      write_channel_intel(chanindemod[1], buf[index1]);
      write_channel_intel(chanindemod[2], buf[index2]);
      write_channel_intel(chanindemod[3], buf[index3]);
      write_channel_intel(chanindemod[4], buf[index4]);
      write_channel_intel(chanindemod[5], buf[index5]);
      write_channel_intel(chanindemod[6], buf[index6]);
      write_channel_intel(chanindemod[7], buf[index7]);
    }
  } // row
}

/* \brief Demodulate the signal and store in DDR
 */
kernel void demodulate(
  global volatile float2 * restrict W,     // pointer to W
  global volatile float2 * restrict dest,  // pointer to dest buffer in DDR
  unsigned int num,                        // signal sample size
  unsigned int count){                     // number of iterations

  float2 bufW[NEAREST_POW_OF_2 + NEAREST_POW_OF_2];

  #pragma unroll 8
  for(unsigned step = 0; step < NEAREST_POW_OF_2; step++){
    bufW[step] = W[step];
  }    

  // Store only the nearest power of 2 number of values
  for(unsigned step = 0; step < count * NEAREST_POW_OF_2 / 8; step++){

    float2 data[8];
    #pragma unroll 8
    for(unsigned i = 0; i < 8; i++){
      data[i] = read_channel_intel(chanindemod[i]);
    }

    // modulate and store
    #pragma unroll 8
    for(unsigned i = 0; i < 8; i++){
      float x = bufW[((step * 8) + i) & (NEAREST_POW_OF_2 - 1)].x;
      float y = bufW[((step * 8) + i) & (NEAREST_POW_OF_2 - 1)].y;
      float a = data[i].x; 
      float b = data[i].y; 

      data[i].x = (x * a) - (y * b);
      data[i].y = (x * b) + (y * a);

      float2 tempzero;
      tempzero.x = 0.0f;
      tempzero.y = 0.0f;

      dest[(step * 8) + i] = ( ((step * 8) + i) & (NEAREST_POW_OF_2 - 1)) < num ? data[i]: tempzero;
    }
  }
}