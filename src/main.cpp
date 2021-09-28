// Arjun Ramaswami
#include <iostream>
#include <math.h>
#include "helper.hpp"
#include "chirpz.hpp"
#include "cstdlib"

using namespace std;

int main(int argc, char* argv[]){

  CONFIG chirpz_config;
  parse_args(argc, argv, chirpz_config);
  print_config(chirpz_config);

  if(chirpz_config.cpuonly){
    cpu_t cpu_timing = {0.0, false};

    switch(chirpz_config.dim){
      case 1:{
        float2 *inp = new float2[chirpz_config.num]();
        float2 *out = new float2[chirpz_config.num]();

        create_data_1d(inp, chirpz_config.num);
        cpu_timing = chirpz_cpu_1d(inp, out, chirpz_config);
        if(cpu_timing.valid == false){
          cout << "Error in CPU Chirp-z Implementation\n";
        }
        bool status = verify_chirp_1d(inp, out, chirpz_config.num);
        if(!status){
          cout << "FFTW and Implementation not the same!" << endl;
        }
        else{
          cout << "-- Verified with FFTW\n";
        }
        free(inp);
        free(out);
        break;
      }
      case 2: {
        cout << "2D Chirp" << endl;
        size_t sz = pow(chirpz_config.num, chirpz_config.dim);
        float2 *inp = new float2[sz]();
        float2 *temp = new float2[sz]();
        float2 *out = new float2[sz]();

        create_data(inp, chirpz_config.num);

        for(size_t i = 0; i < chirpz_config.num; i++){
          cpu_timing = chirpz_cpu_1d(&inp[i * chirpz_config.num], &temp[i*chirpz_config.num], chirpz_config);
          if(cpu_timing.valid == false){
            cout << "Error in CPU Chirp-z Implementation\n";
          }
        }
        // Transpose
        //transpose(temp, chirpz_config);

        float2 *tmp = new float2[sz]();
        for(size_t i = 0; i < chirpz_config.num; i++){
          for(size_t j = 0; j < chirpz_config.num; j++){
            tmp[(i * chirpz_config.num) + j].x = temp[(j * chirpz_config.num) + i].x;
            tmp[(i * chirpz_config.num) + j].y = temp[(j * chirpz_config.num) + i].y;
          }
        }

        for(size_t i = 0; i < chirpz_config.num * chirpz_config.num; i++){
          temp[i].x = tmp[i].x;
          temp[i].y = tmp[i].y;
        }

        // Column wise Chirp
        for(size_t i = 0; i < chirpz_config.num; i++){
          cpu_timing = chirpz_cpu_1d(&temp[i * chirpz_config.num], &out[i*chirpz_config.num], chirpz_config);
          if(cpu_timing.valid == false){
            cout << "Error in CPU Chirp-z Implementation\n";
          }
        }
        // Transpose back
        for(size_t i = 0; i < chirpz_config.num; i++){
          for(size_t j = 0; j < chirpz_config.num; j++){
            tmp[(i * chirpz_config.num) + j].x = out[(j * chirpz_config.num) + i].x;
            tmp[(i * chirpz_config.num) + j].y = out[(j * chirpz_config.num) + i].y;
          }
        }

        for(size_t i = 0; i < chirpz_config.num * chirpz_config.num; i++){
          out[i].x = tmp[i].x;
          out[i].y = tmp[i].y;
        }

        /*
        cpu_timing = chirpz_cpu_2d(inp, out, chirpz_config);
        if(cpu_timing.valid == false){
          cout << "Error in CPU Chirp-z Implementation\n";
        }
        */
        bool status = verify_chirp_2d(inp, out, chirpz_config.num);
        if(!status){
          cout << "FFTW and Implementation not the same!" << endl;
        }
        else{
          cout << "-- Verified with FFTW\n";
        }
        free(inp);
        free(temp);
        free(tmp);
        free(out);
        break;
      }
      case 3:{
        cout << "3D Chirp" << endl;
        size_t sz = pow(chirpz_config.num, chirpz_config.dim);
        float2 *inp = new float2[sz]();
        float2 *out = new float2[sz]();
        free(inp);
        free(out);
        break;
      }
      default:{
        cout << "Default" << endl;
        break;
      }
    }

    disp_results(chirpz_config, cpu_timing); 
    return EXIT_SUCCESS;
  }

  return EXIT_SUCCESS;
}