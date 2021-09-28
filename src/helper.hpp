#ifndef HELPER_HPP
#define HELPER_HPP

#include "chirpz.hpp"

void parse_args(int argc, char* argv[], CONFIG &config);

void print_config(CONFIG config);

//void disp_results(CONFIG config, fpga_t fpga_timing, double api_t);

void disp_results(CONFIG config, cpu_t cpu_timing);

double getTimeinMilliSec();

unsigned next_second_power_of_two(unsigned x);

void create_data(float2 *inp, unsigned num);
void create_data_1d(float2 *inp, unsigned num);

#endif // HELPER_HPP
