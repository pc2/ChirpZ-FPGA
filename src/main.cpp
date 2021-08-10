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

  float2 *inp = new float2[chirpz_config.num]();
  float2 *out = new float2[chirpz_config.num]();

  create_data_1d(inp, chirpz_config.num);

  if(chirpz_config.cpuonly){
    cpu_t cpu_timing = {0.0, false};

    if(chirpz_config.dim == 1){
      cpu_timing = chirpz_cpu_1d(inp, out, chirpz_config);
    }
    if(cpu_timing.valid == false){
      cout << "Error in CPU Chirp-z Implementation\n";
      return EXIT_FAILURE;
    }
    cout << "-- FFT successful" << endl;

    bool status = verify_chirp_1d(inp, out, chirpz_config.num);
    if(!status){
      cout << "FFTW and Implementation not the same!" << endl;
    }
    else{
      cout << "-- Verified with FFTW\n";
    }

    disp_results(chirpz_config, cpu_timing); 
    return EXIT_SUCCESS;
  }

  free(inp);
  free(out);

  return EXIT_SUCCESS;
}