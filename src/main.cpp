// Arjun Ramaswami
#include <iostream>
#include "helper.hpp"
#include "chirpz.hpp"
#include "cstdlib"

using namespace std;

int main(int argc, char* argv[]){

  //std::string inp_fname, out_fname;
  CONFIG chirpz_config;
  parse_args(argc, argv, chirpz_config);
  print_config(chirpz_config);

  if(chirpz_config.cpuonly){
#ifdef USE_FFTW
    cpu_t cpu_timing = {0.0, false};
    cpu_timing = chirpz_cpu_1d(chirpz_config);
    if(cpu_timing.valid == false){
      cout << "Error in CPU Chirp-z Implementation\n";
      return EXIT_FAILURE;
    }
    cout << "FFT successful" << endl;
    disp_results(chirpz_config, cpu_timing); 
    return EXIT_SUCCESS;
#else
    cerr << "FFTW not found" << endl;
    return EXIT_FAILURE;
#endif
  }

  return EXIT_SUCCESS;
}