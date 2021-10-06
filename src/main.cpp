// Arjun Ramaswami
#include <iostream>
#include <math.h>
#include "helper.hpp"
#include "chirpz.hpp"
#include "cstdlib"
#include <vector>
using namespace std;

int main(int argc, char* argv[]){

  CONFIG chirpz_config;
  parse_args(argc, argv, chirpz_config);
  print_config(chirpz_config);

  bool status = false;
  const unsigned num = chirpz_config.num;
  const unsigned sz = pow(num, chirpz_config.dim);

  vector<float2> inp(sz, {0.0, 0.0});
  vector<float2> out(sz, {0.0, 0.0});

  create_data(inp.data(), sz);

  try{
    if(chirpz_config.cpuonly){
      switch(chirpz_config.dim){
        case 1:{
          cout << "-- 1D Chirp" << endl;
          cout << "-- Executing ...\n";
          chirpz1d_cpu(inp.data(), out.data(), num);
          cout << "-- Verifying ...\n";
          status = verify_chirp1d(inp.data(), out.data(), num);
          break;
        }
        case 2: {
          cout << "-- 2D Chirp" << endl;
          cout << "-- Executing ...\n";
          chirpz2d_cpu(inp.data(), out.data(), num);
          cout << "-- Verifying ...\n";
          status = verify_chirp2d(inp.data(), out.data(), num);
          break;
        }
        case 3:{
          cout << "3D Chirp" << endl;
          cout << "-- Executing ...\n";
          chirpz3d_cpu(inp.data(), out.data(), num);
          status = verify_chirp3d(inp.data(), out.data(), num);
          break;
        }
        default:{
          cout << "Choose a dimension!" << endl;
          break;
        }
      }
    }
    else{cout << "FPGA impl impending\n";}
  }
  catch(const char* msg){
    cerr << msg << endl;
  }

  if(!status){ cout << "-- FFTW and Implementation not the same!" << endl;}
  else{ cout << "-- Works and verified using FFTW\n\n"; }

  return EXIT_SUCCESS;
}