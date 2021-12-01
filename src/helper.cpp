//  Author: Arjun Ramaswami
#include <iostream>
#include <iomanip>
#include <fftw3.h>
#include <math.h>
#include "cxxopts.hpp"
#include "helper.hpp"
#include "config.h"
#include "chirpz.hpp"
using namespace std;

/**
 * \brief  create random single precision complex floating point values  
 * \param  inp : pointer to float2 data of size N 
 * \param  num   : number of points in the array
 * \return true if successful
 */
void create_data(float2 *inp, const unsigned num){
  if(inp == NULL || num <= 0){ throw "Bad args in create data function";}

  for(size_t i = 0; i < num; i++){
    inp[i].x = (float)((float)rand() / (float)RAND_MAX);
    inp[i].y = (float)((float)rand() / (float)RAND_MAX);
  }
}

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
 * \brief print time taken for fpga and fftw runs
 * \param config: custom structure of variables storing config values 
 * \param runtime: iteration number of fpga timing measurements
 * \param total_api_time: time taken to call iter times the host code
 */
void perf_measures(const CONFIG config, fpga_t *runtime){

  fpga_t avg_runtime = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  for(unsigned i = 0; i < config.iter; i++){
    avg_runtime.exec_t += runtime[i].exec_t;
    avg_runtime.pcie_read_t += runtime[i].pcie_read_t;
    avg_runtime.pcie_write_t += runtime[i].pcie_write_t;
  }
  avg_runtime.exec_t = avg_runtime.exec_t / config.iter;
  avg_runtime.pcie_read_t = avg_runtime.pcie_read_t / config.iter;
  avg_runtime.pcie_write_t = avg_runtime.pcie_write_t / config.iter;

  fpga_t variance = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  fpga_t sd = {0.0, 0.0, 0.0, 0.0, 0.0, 0};
  for(unsigned i = 0; i < config.iter; i++){
    variance.exec_t += pow(runtime[i].exec_t - avg_runtime.exec_t, 2);
    variance.pcie_read_t += pow(runtime[i].pcie_read_t - avg_runtime.pcie_read_t, 2);
    variance.pcie_write_t += pow(runtime[i].pcie_write_t - avg_runtime.pcie_write_t, 2);
  }
  sd.exec_t = sqrt(variance.exec_t / config.iter);
  sd.pcie_read_t = sqrt(variance.pcie_read_t / config.iter);
  sd.pcie_write_t = sqrt(variance.pcie_write_t / config.iter);

  double avg_total_runtime = avg_runtime.exec_t + avg_runtime.pcie_write_t + avg_runtime.pcie_read_t;

  double gpoints_per_sec = (config.batch * pow(config.num, config.dim)) / (avg_runtime.exec_t * 1e-3 * 1024 * 1024);

  double gBytes_per_sec = gpoints_per_sec * 8; // bytes

  double gflops = config.batch * config.dim * 5 * pow(config.num, config.dim) * (log((double)config.num)/log((double)2))/(avg_runtime.exec_t * 1e-3 * 1024*1024*1024); 

  printf("\n\n------------------------------------------\n");
  printf("Measurements \n");
  printf("--------------------------------------------\n");
  printf("%s", config.iter>1 ? "Average Measurements of iterations\n":"");
  printf("PCIe Write          = %.4lfms\n", avg_runtime.pcie_write_t);
  printf("Kernel Execution    = %.4lfms\n", avg_runtime.exec_t);
  printf("Kernel Exec/Batch   = %.4lfms\n", avg_runtime.exec_t / config.batch);
  printf("PCIe Read           = %.4lfms\n", avg_runtime.pcie_read_t);
  printf("Total               = %.4lfms\n", avg_total_runtime);
  printf("Throughput          = %.4lfGFLOPS/s | %.4lf GB/s\n", gflops, gBytes_per_sec);
  if(config.iter > 1){
    printf("\n");
    printf("%s", config.iter>1 ? "Deviation of runtimes among iterations\n":"");
    printf("PCIe Write          = %.4lfms\n", sd.pcie_write_t);
    printf("Kernel Execution    = %.4lfms\n", sd.exec_t);
    printf("PCIe Read           = %.4lfms\n", sd.pcie_read_t);
  }
}

void parse_args(int argc, char* argv[], CONFIG &config){

  try{
    cxxopts::Options options("Chirp-Z", "Chirp-Z Filter for non-powers-of-2 3D FFT");
    options.add_options()
      ("n, num", "Size of FFT dim", cxxopts::value<unsigned>()->default_value("31"))
      ("d, dim", "Number of FFT dim", cxxopts::value<unsigned>()->default_value("3"))
      ("p, path", "Path to bitstream", cxxopts::value<string>())
      ("c, cpu-only", "CPU FFTW Only", cxxopts::value<bool>()->default_value("false") )
      ("i, iter", "Number of iterations", cxxopts::value<unsigned>()->default_value("1"))
      ("y, noverify", "No verification", cxxopts::value<bool>()->default_value("false") )
      ("b, batch", "Num of even batches", cxxopts::value<unsigned>()->default_value("1") )
      ("s, use_svm", "SVM enabled", cxxopts::value<bool>()->default_value("false") )
      ("e, emulate", "toggle emulation", cxxopts::value<bool>()->default_value("false") )
      ("k, back", "Toggle Backward FFT", cxxopts::value<bool>()->default_value("false") )
      ("h,help", "Print usage")
    ;
    auto opt = options.parse(argc, argv);

    // print help
    if (opt.count("help")){
      cout << options.help() << endl;
      exit(0);
    }

    config.cpuonly = opt["cpu-only"].as<bool>();
    if( (opt.count("path") && config.cpuonly))
      throw "\tRun either cpu or emulation fpga";
    
    if(!config.cpuonly){
      if(opt.count("path"))
        config.path = opt["path"].as<string>();
      else
        throw "\tPlease input path to bitstream\n";
    }

    config.dim = opt["dim"].as<unsigned>();
    config.num = opt["num"].as<unsigned>();
    config.iter = opt["iter"].as<unsigned>();
    config.batch = opt["batch"].as<unsigned>();
    config.inv = opt["back"].as<bool>();

    if(config.cpuonly && config.batch > 1 && config.dim == 1){
      throw "Batched CPU ChirpZ not implemented, only FPGA\n";
    }
    config.noverify = opt["noverify"].as<bool>();
    config.use_svm = opt["use_svm"].as<bool>();
    config.emulate = opt["emulate"].as<bool>();
  }
  catch(const char* msg){
    cerr << msg << endl;
    exit(1);
  }
}

void print_config(CONFIG config, const char* platform_name){
  cout << endl;
  cout << "CONFIGURATION: \n";
  cout << "---------------\n";
  printf("Type        : Complex to Complex %s Transform\n", config.inv == false ? "" : "Backward");
  printf("Points      : %d%s \n", config.num, config.dim == 1 ? "" : config.dim == 2 ? "^2" : "^3");
  cout <<"Bitstream   : " << config.path << endl;
  cout <<"Iterations  : "<< config.iter << endl;
  cout <<"Batch       : "<< config.batch << endl;
  printf("Emulation   : %s\n", config.emulate == 1 ? "Yes": "No");
  printf("Platform    : %s\n", platform_name);
  cout <<"----------------\n\n";
}

unsigned next_second_power_of_two(unsigned x) {
  x = x - 1; 
  x = x | (x >> 1); 
  x = x | (x >> 2); 
  x = x | (x >> 4); 
  x = x | (x >> 8); 
  x = (x + 1) << 1;
  return x;
} 