#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

const unsigned next_second_power_of_two(unsigned x) {
  x = x - 1; 
  x = x | (x >> 1); 
  x = x | (x >> 2); 
  x = x | (x >> 4); 
  x = x | (x >> 8); 
  x = (x + 1) << 1;
  return x;
} 

// Chirp Z implementation
cpu_t chirpz_cpu(struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};

  //unsigned chirp_num = next_second_power_of_two(config.num);

  timing_cpu.valid = true;
  return timing_cpu;
}

