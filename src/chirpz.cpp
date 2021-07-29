#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

// Chirp Z implementation
cpu_t chirpz_cpu(struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};

  timing_cpu.valid = true;
  return timing_cpu;
}

