#include <iostream>
#include <fstream>
#include <omp.h>
#include <fftw3.h>
#include <math.h>

#include "chirpz.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

static fftwf_plan plan_fftwf, plan_chirp_sig, plan_chirp_filter, plan_inv_chirp;

void cleanup_plans(){
  fftwf_destroy_plan(plan_fftwf);
  fftwf_destroy_plan(plan_chirp_sig);
  fftwf_destroy_plan(plan_chirp_filter);
  fftwf_destroy_plan(plan_inv_chirp);
}

bool create_data(fftwf_complex *fftw_verify, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  if(fftw_verify == NULL || num <= 0){
    return false;
  }

  if(chirp_sig == NULL || chirp_filter == NULL || chirp_num <= 0){
    return false;
  }

  for(size_t i = 0; i < chirp_num; i++){
    for(size_t j = 0; j < chirp_num; j++){
      for(size_t k = 0; k < chirp_num; k++){
          
        size_t index = (i*chirp_num*chirp_num) + (j*chirp_num) + k;
        if(k < num){
          chirp_sig[index][0] = k;
          chirp_sig[index][1] = 0.0;
          //chirp_sig[index][0] = (float)((float)rand() / (float)RAND_MAX);
          //chirp_sig[index][1] = (float)((float)rand() / (float)RAND_MAX);
        }
        else{
          chirp_sig[index][0] = 0.0f;
          chirp_sig[index][1] = 0.0f;
        }
        chirp_filter[index][0] = 0.0f;
        chirp_filter[index][1] = 0.0f;

        //cout << index << ": (" << chirp_sig[index][0] << ", " << chirp_sig[index][1] << ") " << endl;
      }
    }
  }
  for(size_t i = 0; i < num; i++){
    for(size_t j = 0; j < num; j++){
      for(size_t k = 0; k < num; k++){
          
        size_t chirp_index = (i*chirp_num*chirp_num) + (j*chirp_num) + k;
        size_t index = (i*num*num) + (j*num) + k;

        fftw_verify[index][0] = chirp_sig[chirp_index][0];
        fftw_verify[index][1] = chirp_sig[chirp_index][1];

        //cout << "Chirp Index: " << chirp_index << " Index " << index <<" k: " << k;
        cout << " Chirp: (" << chirp_sig[chirp_index][0] << ", " << chirp_sig[chirp_index][1] << ") - Index: (" << fftw_verify[index][0] << "," << fftw_verify[index][1] << ") " << endl;

      }
    }
  }

  return true;
}

bool modulate(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  //unsigned num_pts = num * num * num;

  for(size_t i = 0; i < num; i++){
    for(size_t j = 0; j < num; j++){
      for(size_t k = 0; k < num; k++){

        float x = cos(M_PI * k * k / num);
        float y = sin(M_PI * k * k / num);

        size_t index = (i*num*num) + (j*num) + k;
        size_t oth_index = (i*num*num) + (j*num) + (num - k);

        chirp_sig[index][0] = (chirp_sig[index][0] * x) + (chirp_sig[index][1] * -1 * y) ;
        chirp_sig[index][1] = (chirp_sig[index][1] * x) - (chirp_sig[index][0] * y);

        chirp_filter[index][0] = x;
        chirp_filter[index][1] = y;

        chirp_filter[oth_index][0] = x;
        chirp_filter[oth_index][1] = y;
      }
    }
  }

  /*
  for(size_t i = 0; i < num_pts; i++){
    unsigned index = i % chirp_num;  
    float x = cos(M_PI * index * index / num);
    float y = sin(M_PI * index * index / num);
    // should this be num instead of chirp_num?

    chirp_sig[i][0] = (chirp_sig[i][0] * x) + (chirp_sig[i][1] * -1 * y) ;
    chirp_sig[i][1] = (chirp_sig[i][1] * x) - (chirp_sig[i][0] * y);
    chirp_filter[i][0] = x;
    chirp_filter[i][1] = y;
  }
  */

  return true;
}

void demodulate(fftwf_complex *chirp_sig, const unsigned num, const unsigned chirp_num){

  unsigned num_pts = num * num * num;
  //unsigned chirp_num_pts = chirp_num * chirp_num * chirp_num;

  for(size_t i = 0; i < num_pts; i++){
    unsigned index = i % num;
    float x = cos(M_PI * index * index / num);
    float y = sin(M_PI * index * index / num);
    chirp_sig[i][0] = ((chirp_sig[i][0] * x) + (chirp_sig[i][1] * y)) / (chirp_num* num);
    chirp_sig[i][1] = ((chirp_sig[i][1] * x) - (chirp_sig[i][0] * y)) / (chirp_num*num);
  }
}

bool verify_chirp(fftwf_complex *chirp_sig, fftwf_complex *fftwf_verify, unsigned num_pts){

  bool flag = true;
  for(size_t i = 0; i < num_pts; i++){
    if( (chirp_sig[i][0] != fftwf_verify[i][0]) || (chirp_sig[i][1] != fftwf_verify[i][1])){
      flag = false; 
      //break;
    }
    cout << "i: " << i << " Sig: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ") " << "FFTW: (" << fftwf_verify[i][0] << ", " << fftwf_verify[i][1] << ") " << endl;
  }
  return flag;
}

// Chirp Z implementation
cpu_t chirpz_cpu(struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};

  unsigned num = config.num;
  size_t num_pts = num * num * num;
  fftwf_complex *fftwf_verify = fftwf_alloc_complex(num_pts);

  cout << "-- FFT Dimensions: "<< num << " x " << num << " x " << num << endl;

  const int dim = 3;
  const int n[3] = {(int)num, (int)num, (int)num};
  int idist = num * num * num, odist = num * num * num;
  int istride = 1, ostride = 1;

  int threads_ok = fftwf_init_threads(); 
  if(threads_ok == 0){
    throw "Something went wrong with Multithreaded FFTW! Exiting... \n";
  }
  
  fftwf_plan_with_nthreads((int)config.threads);

  cout << "-- Creating Plan" << endl;
  unsigned fftw_plan = FFTW_PLAN;
  switch(fftw_plan){
    case FFTW_MEASURE:  cout << "-- FFTW Plan: Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "-- FFTW Plan: Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "-- FFTW Plan: Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "-- FFTW Plan: Exhaustive\n";
                        break;
    default: throw "-- Incorrect plan set\n";
            break;
  }

  int wis_status = fftwf_import_wisdom_from_filename(config.wisdomfile.c_str());
  if(wis_status == 0){
    cout << "-- Cannot import wisdom for FFTW verification from " << config.wisdomfile << endl;
  }
  else{
    cout << "-- Importing wisdom FFTW verification from " << config.wisdomfile << endl;
    fftw_plan = FFTW_WISDOM_ONLY | FFTW_ESTIMATE;
  }

  plan_fftwf = fftwf_plan_many_dft(dim, n, 1, fftwf_verify, NULL, istride, idist, fftwf_verify, NULL, ostride, odist, FFTW_FORWARD, fftw_plan);
  
  unsigned chirp_num = next_second_power_of_two(config.num);
  size_t chirp_num_pts = chirp_num * chirp_num * chirp_num;
  fftwf_complex *chirp_sig = fftwf_alloc_complex(chirp_num_pts);
  fftwf_complex *chirp_filter = fftwf_alloc_complex(chirp_num_pts);

  cout << "-- Chirp Dimensions: "<< chirp_num << " x " << chirp_num << " x " << chirp_num << endl << endl;

  const int chirp_n[3] = {(int)chirp_num, (int)chirp_num, (int)chirp_num};
  int chirp_idist = chirp_num_pts, chirp_odist = chirp_num_pts;

  unsigned fftw_chirp_plan = FFTW_PLAN;

  int chirp_wis_status = fftwf_import_wisdom_from_filename(config.chirp_wisdomfile.c_str());
  if(chirp_wis_status == 0){
    cout << "-- Cannot import wisdom for Chirp DFT from " << config.chirp_wisdomfile << endl;
  }
  else{
    cout << "-- Importing wisdom for Chirp DFT from " << config.chirp_wisdomfile << endl;
    fftw_chirp_plan = FFTW_WISDOM_ONLY | FFTW_ESTIMATE;
  }

  double plan_start = getTimeinMilliSec();
  plan_chirp_sig = fftwf_plan_many_dft(dim, chirp_n, 1, chirp_sig, NULL, istride, chirp_idist, chirp_sig, NULL, ostride, chirp_odist, FFTW_FORWARD, fftw_chirp_plan);

  plan_chirp_filter = fftwf_plan_many_dft(dim, chirp_n, 1, chirp_filter, NULL, istride, chirp_idist, chirp_filter, NULL, ostride, chirp_odist, FFTW_FORWARD, fftw_chirp_plan);

  plan_inv_chirp = fftwf_plan_many_dft(dim, chirp_n, 1, chirp_sig, NULL, istride, chirp_idist, chirp_sig, NULL, ostride, chirp_odist, FFTW_BACKWARD, fftw_chirp_plan);
  double plan_time = getTimeinMilliSec() - plan_start;
  cout << "-- Time to Plan: " << plan_time << endl;

  if(wis_status == 0){
    int exp_stat = fftwf_export_wisdom_to_filename(config.chirp_wisdomfile.c_str()); 
    if(exp_stat == 0){
      cout << "-- Could not export wisdom file for Chirp DFT to " << config.chirp_wisdomfile.c_str() << endl;
    }
    else{
      cout << "-- Exporting wisdom file to " << config.chirp_wisdomfile.c_str() << endl;
    }
  }

  bool status = create_data(fftwf_verify, chirp_sig, chirp_filter, num, chirp_num);
  if(!status){
    cerr << "Error in Data Creation" << endl;
    fftwf_free(fftwf_verify);
    fftwf_free(chirp_sig);
    fftwf_free(chirp_filter);
    timing_cpu.valid = false;
    return timing_cpu;
  }

  status = modulate(chirp_sig, chirp_filter, num, chirp_num);
  if(!status){
    cerr << "Error in Data Creation" << endl;
    fftwf_free(fftwf_verify);
    fftwf_free(chirp_sig);
    fftwf_free(chirp_filter);
    timing_cpu.valid = false;
    return timing_cpu;
  } 

  fftwf_execute(plan_chirp_filter);
  fftwf_execute(plan_chirp_sig);

  #pragma omp parallel for num_threads(config.threads)
  for(unsigned i = 0; i < chirp_num_pts; i++){
    float x, y;
    x = (chirp_sig[i][0] * chirp_filter[i][0]) - (chirp_sig[i][1] * chirp_filter[i][1]);
    y = (chirp_sig[i][0] * chirp_filter[i][1]) + (chirp_sig[i][1] * chirp_filter[i][0]);

    chirp_sig[i][0] = x;
    chirp_sig[i][1] = y;
  }

  fftwf_execute(plan_inv_chirp);
  demodulate(chirp_sig, num, chirp_num);

  fftwf_execute(plan_fftwf);

  status = verify_chirp(chirp_sig, fftwf_verify, num_pts);
  if(!status){
    cerr << "FFTW and Chirp DFT not same" << endl;
    fftwf_free(fftwf_verify);
    fftwf_free(chirp_sig);
    fftwf_free(chirp_filter);
    timing_cpu.valid = false;
    return timing_cpu;
  }   

  fftwf_free(fftwf_verify);
  fftwf_free(chirp_sig);
  fftwf_free(chirp_filter);
  cleanup_plans();

  timing_cpu.valid = true;
  return timing_cpu;
}

