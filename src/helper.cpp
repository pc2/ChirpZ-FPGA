//  Author: Arjun Ramaswami
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include "cxxopts.hpp"
#include "helper.hpp"
#include "config.h"

using namespace std;

/**
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

/**
 * \brief  print time taken for 3d fft execution and data transfer
 * \param  exec_time    : average time in seconds to execute a parallel 3d FFT
 * \param  gather_time  : average time in seconds to gather results to the master node after transformation
 * \param  flops        : fftw_flops 
 * \param  N1, N2, N3   : fft size
 * \param  nprocs       : number of processes used
 * \param  nthreads     : number of threads used
 * \param  iter         : number of iterations
 * \return true if successful, false otherwise
 */
bool print_results(double exec_time, double gather_time, double flops, unsigned N, unsigned nprocs, unsigned nthreads, unsigned iter, unsigned how_many){

  if(exec_time == 0.0)
    throw "Error in Run\n";
  
  double avg_exec = exec_time / iter;

  cout << "\nMeasurements\n" << "--------------------------\n";
  cout << "Processes           : " << nprocs << endl;
  cout << "Threads             : " << nthreads << endl;
  cout << "FFT Size            : " << N << "^3\n";
  cout << "Batch               : " << how_many << endl;
  cout << "Iterations          : " << iter << endl;
  cout << "Avg Tot Runtime     : " << setprecision(4) << avg_exec << " ms\n";
  cout << "Runtime per batch   : " << (avg_exec / how_many) << " ms\n";
  cout << "Throughput          : " << (flops * 1e-9) << " GFLOPs\n";
  cout << "Time to Transfer    : " << gather_time << "ms\n";
  cout << "--------------------------\n";

  return true;
}

void parse_args(int argc, char* argv[], CONFIG &config){

  try{
    cxxopts::Options options("Chirp-Z", "Chirp-z Filter for non-powers-of-2 3D FFT");
    options.add_options()
      ("p, path", "Path to bitstream", cxxopts::value<string>())
      ("w, wisdomfile", "File to wisdom", cxxopts::value<string>()->default_value("a.out"))
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("64"))
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("t, threads", "Number of threads", cxxopts::value<unsigned>()->default_value("1"))
      ("y, noverify", "No verification", cxxopts::value<bool>()->default_value("false") )
      ("b, batch", "Num of even batches", cxxopts::value<unsigned>()->default_value("1") )
      ("c, cpu-only", "CPU FFTW Only", cxxopts::value<bool>()->default_value("false") )
      ("s, usesvm", "SVM enabled", cxxopts::value<bool>()->default_value("false") )
      ("h,help", "Print usage")
    ;
    auto opt = options.parse(argc, argv);

    // print help
    if (opt.count("help")){
      cout << options.help() << endl;
      exit(0);
    }

    config.cpuonly = opt["cpu-only"].as<bool>();
    if(!config.cpuonly){
      if(opt.count("path")){
        config.path = opt["path"].as<string>();
      }
      else{
        cout << "\tPlease input path to bitstream" << endl;
        exit(1);
      }
    }
    if(opt.count("wisdomfile")){
      config.wisdomfile = opt["wisdomfile"].as<string>();
    }

    config.num = opt["num"].as<unsigned>();
    config.threads = opt["threads"].as<unsigned>();
    config.iter = opt["iter"].as<unsigned>();
    config.batch = opt["batch"].as<unsigned>();
    config.noverify = opt["noverify"].as<bool>();
    config.usesvm = opt["usesvm"].as<bool>();
  }
  catch(const cxxopts::OptionException& e){
    cerr << "Error parsing options: " << e.what() << endl;
    exit(1);
  }
}

void print_config(CONFIG config){
  cout << endl;
  cout << "CONFIGURATION: \n";
  cout << "---------------\n";
  cout << "Bitstream    = " << config.path << endl;
  cout << "Points       = {"<< config.num << ", " << config.num << ", " << config.num << "}" << endl;
  cout << "Wisdom Path  = " << config.wisdomfile << endl;
  switch(FFTW_PLAN){
    case FFTW_MEASURE:  cout << "FFTW Plan    = Measure\n";
                        break;
    case FFTW_ESTIMATE: cout << "FFTW Plan    = Estimate\n";
                        break;
    case FFTW_PATIENT:  cout << "FFTW Plan    = Patient\n";
                        break;
    case FFTW_EXHAUSTIVE: cout << "FFTW Plan   = Exhaustive\n";
                        break;
    default: throw "-- Incorrect plan set\n";
            break;
  }

  cout << "Threads      = "<< config.threads << endl;
  cout << "Iterations   = "<< config.iter << endl;
  cout << "Batch        = "<< config.batch << endl;
  cout << "----------------\n\n";
}

/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliseconds(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

void disp_results(CONFIG config, cpu_t timing_cpu){

  cout << endl << endl;
  cout << "MEASUREMENTS \n";
  cout << "--------------\n";
  cout << "Points           : " << config.num << "^3\n";
  cout << "Threads          : " << config.threads << endl;
  cout << "Iterations       : " << config.iter << endl << endl;

  cout << "CPU:" << endl;
  cout << "----" << endl;
  cout << "Chirp-Z Runtime   : "<< timing_cpu.chirpz_t << "ms" << endl;
}