// Arjun Ramaswami
#include <iostream>
#include <math.h>
#include <vector>
#include "cstdlib"
#include "helper.hpp"
#include "chirpz.hpp"
#include "fftfpga/fftfpga.h"

using namespace std;

int main(int argc, char* argv[]){

  CONFIG chirpz_config;
  parse_args(argc, argv, chirpz_config);

  const char* platform;
  if(chirpz_config.emulate)
    platform = "Intel(R) FPGA Emulation Platform for OpenCL(TM)";
  else
    platform = "Intel(R) FPGA SDK for OpenCL(TM)";
  
  print_config(chirpz_config, platform);

  bool status = false;
  const unsigned num = chirpz_config.num;
  const unsigned sz = pow(num, chirpz_config.dim) * chirpz_config.batch;

  vector<float2> inp(sz, {0.0, 0.0});
  vector<float2> out(sz, {0.0, 0.0});
  fpga_t runtime[chirpz_config.iter];

  create_data(inp.data(), sz);

  int isInit;
  if(!chirpz_config.cpuonly){
    isInit = fpga_initialize(platform, chirpz_config.path.data(), chirpz_config.use_svm);
    if(isInit != 0){
      cerr << "FPGA initialization error\n";
      return EXIT_FAILURE;
    }
  }

  try{
    for(unsigned i = 0; i < chirpz_config.iter; i++){
      if(chirpz_config.cpuonly){
        switch(chirpz_config.dim){
          case 1:{
            cout << "-- 1D Chirp" << endl;
            cout << "-- Executing ...\n";
            chirpz1d_cpu(inp.data(), out.data(), num, chirpz_config.inv);
            cout << "-- Verifying ...\n";
            status = verify_chirp1d(inp, out, num, 1, chirpz_config.inv);
            break;
          }
          case 2: {
            cout << "-- 2D Chirp" << endl;
            cout << "-- Executing ...\n";
            chirpz2d_cpu(inp.data(), out.data(), num, chirpz_config.inv);
            cout << "-- Verifying ...\n";
            status = verify_chirp2d(inp.data(), out.data(), num);
            break;
          }
          case 3:{
            cout << "3D Chirp" << endl;
            cout << "-- Executing ...\n";
            chirpz3d_cpu(inp.data(), out.data(), num, chirpz_config.inv);
            status = verify_chirp3d(inp.data(), out.data(), num);
            break;
          }
          default:{
            cout << "Choose a dimension!" << endl;
            break;
          }
        }
      }
      else{
        switch(chirpz_config.dim){
          case 1:{
            printf("Iteration %u\n-- 1D Chirp\n", i);
            runtime[i] = fftfpgaf_c2c_chirp1d(num, inp.data(), out.data(), chirpz_config.inv, chirpz_config.batch);
            cout << "-- Verifying ...\n";
            status = verify_chirp1d(inp, out, num, chirpz_config.batch, chirpz_config.inv);
            break;
          }
          default:{
            cout << "Other dimensions are not implemented yet" << endl;
            break;
          }
        }
      }
    }
  }
  catch(const char* msg){
    cerr << msg << endl;
    if(!chirpz_config.cpuonly)
      fpga_final();
    return EXIT_FAILURE;
  }
  if(!chirpz_config.cpuonly)
    fpga_final();

  if(!status){ cout << "-- FFTW and Implementation not the same!" << endl;}
  else{ 
    cout << "-- Works and verified using FFTW\n"; 
    if(!chirpz_config.cpuonly)
      perf_measures(chirpz_config, runtime);
  }

  return EXIT_SUCCESS;
}