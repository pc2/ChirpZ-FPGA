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

static void cleanup_plans_1d(){
  fftwf_destroy_plan(plan_fftwf);
  fftwf_destroy_plan(plan_chirp_sig);
  fftwf_destroy_plan(plan_chirp_filter);
  fftwf_destroy_plan(plan_inv_chirp);
}

static bool create_data_1d(fftwf_complex *fftw_verify, fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  if(fftw_verify == NULL || num <= 0 || chirp_sig == NULL || chirp_filter == NULL || chirp_num <= 0){
    return false;
  }

  for(size_t i = 0; i < chirp_num; i++){
    chirp_sig[i][0] = 0.0f;
    chirp_sig[i][1] = 0.0f;
    chirp_filter[i][0] = 0.0f;
    chirp_filter[i][1] = 0.0f;
    fftw_verify[i][0] = 0.0f;
    fftw_verify[i][1] = 0.0f;
  }

#ifdef NDEBUG
  cout << "Signal: Data Creation\n";
  for(size_t i = 0; i < num; i++){
    fftw_verify[i][0] = chirp_sig[i][0] = (float)i;

    cout << i << ": Chirp: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ")\n";
  }
  cout << endl;
#endif

  return true;
}

static void modulate_1d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned num, const unsigned chirp_num){

  for(size_t i = 0; i < num; i++){
    float x = cos(M_PI * i * i / num);
    float y = sin(M_PI * i * i / num);
    float a = chirp_sig[i][0]; 
    float b = chirp_sig[i][1]; 

    chirp_sig[i][0] = (x * b) + (y * a);
    chirp_sig[i][1] = (x * a) - (y * b);

    chirp_filter[i][0] = x;
    chirp_filter[i][1] = y;

    chirp_filter[chirp_num -i][0] = x;
    chirp_filter[chirp_num -i][1] = y;
  }

  /*
  cout << "Signal before FFT" << endl;
  for(size_t i = 0; i < chirp_num; i++){
    cout << i << ": Chirp: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ") \n";
  }
  cout << endl << endl;
  */
}

static void point_mult_1d(fftwf_complex *chirp_sig, fftwf_complex *chirp_filter, const unsigned chirp_num){

  for(unsigned i = 0; i < chirp_num; i++){
    float x, y;
    x = (chirp_sig[i][0] * chirp_filter[i][0]) - (chirp_sig[i][1] * chirp_filter[i][1]);
    y = (chirp_sig[i][0] * chirp_filter[i][1]) + (chirp_sig[i][1] * chirp_filter[i][0]);

    chirp_sig[i][0] = x;
    chirp_sig[i][1] = y;

  }
  cout << endl << endl;
}

static void demodulate_1d(fftwf_complex *chirp_sig, const unsigned chirp_num, const unsigned num){

  for(size_t i = 0; i < num; i++){
    float x = cos(M_PI * i * i / num);
    float y = sin(M_PI * i * i / num);
    float a = chirp_sig[i][0];
    float b = chirp_sig[i][1];

    chirp_sig[i][1] = -1 * ((x * a) + (y * b)) / (chirp_num);
    chirp_sig[i][0] = ((x * b) - (y * a)) / (chirp_num);
  }
}

static bool verify_chirp_1d(fftwf_complex *chirp_sig, fftwf_complex *fftwf_verify, const unsigned num){

  bool flag = true;
  for(size_t i = 0; i < num; i++){
    if( (chirp_sig[i][0] != fftwf_verify[i][0]) || (chirp_sig[i][1] != fftwf_verify[i][1])){
      flag = false; 
    }
  }
  return flag;
}


// Chirp Z implementation
cpu_t chirpz_cpu_1d(struct CONFIG& config){

  cpu_t timing_cpu = {0.0, false};

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

  const unsigned num = config.num;
  cout << "-- FFT Dimensions: "<< num << endl;
  fftwf_complex *fftwf_verify = fftwf_alloc_complex(num);
  plan_fftwf = fftwf_plan_dft_1d(num, fftwf_verify, fftwf_verify, FFTW_FORWARD, fftw_plan);
  
  const unsigned chirp_num = next_second_power_of_two(config.num);
  cout << "-- Chirp Dimensions: "<< chirp_num << endl;
  fftwf_complex *chirp_sig = fftwf_alloc_complex(chirp_num);
  fftwf_complex *chirp_filter = fftwf_alloc_complex(chirp_num);

  plan_chirp_sig = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_FORWARD, fftw_plan);
  plan_chirp_filter = fftwf_plan_dft_1d(chirp_num, chirp_filter, chirp_filter, FFTW_FORWARD, fftw_plan);
  plan_inv_chirp = fftwf_plan_dft_1d(chirp_num, chirp_sig, chirp_sig, FFTW_BACKWARD, fftw_plan);

  bool status = create_data_1d(fftwf_verify, chirp_sig, chirp_filter, num, chirp_num);
  if(!status){
    cerr << "Error in Data Creation" << endl;
    fftwf_free(fftwf_verify);
    fftwf_free(chirp_sig);
    fftwf_free(chirp_filter);
    timing_cpu.valid = false;
    return timing_cpu;
  }

  modulate_1d(chirp_sig, chirp_filter, num, chirp_num);

  fftwf_execute(plan_chirp_filter);
  fftwf_execute(plan_chirp_sig);

#ifdef NDEBUG
  cout << "After both FFT" << endl;
  for(size_t i = 0; i < chirp_num; i++){
    cout << i << ": Chirp: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ") - Filter: (" << chirp_filter[i][0] << ", " << chirp_filter[i][1] << ")\n";
  }
  cout << endl << endl;
#endif

  point_mult_1d(chirp_sig, chirp_filter, chirp_num);

  fftwf_execute(plan_inv_chirp);

#ifdef NDEBUG
  cout << "After IFFT" << endl;
  for(size_t i = 0; i < chirp_num; i++){
    cout << i << ": Chirp: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ") - Filter: (" << chirp_filter[i][0] << ", " << chirp_filter[i][1] << ")\n";
  }
#endif

  demodulate_1d(chirp_sig, chirp_num, num);

#ifdef NDEBUG
  cout << "After demodulation" << endl;
  for(size_t i = 0; i < chirp_num; i++){
    cout << i << ": Chirp: (" << chirp_sig[i][0] << ", " << chirp_sig[i][1] << ") - Filter: (" << chirp_filter[i][0] << ", " << chirp_filter[i][1] << ")\n";
  }
#endif
  
  fftwf_execute(plan_fftwf);

  status = verify_chirp_1d(chirp_sig, fftwf_verify, num);
  if(!status){
    cerr << "FFTW and Chirp DFT not same" << endl;
    fftwf_free(fftwf_verify);
    fftwf_free(chirp_sig);
    fftwf_free(chirp_filter);
    timing_cpu.valid = false;
    return timing_cpu;
  }   

  fftwf_free(chirp_sig);
  fftwf_free(chirp_filter);
  cleanup_plans_1d();

  //fftwf_free(fftwf_verify);
  timing_cpu.valid = true;
  return timing_cpu;
}

