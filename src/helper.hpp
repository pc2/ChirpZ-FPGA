#ifndef HELPER_HPP
#define HELPER_HPP

#include "chirpz.hpp"

void parse_args(int argc, char* argv[], CONFIG &config);

void print_config(CONFIG config);

unsigned next_second_power_of_two(unsigned x);

void create_data(float2 *inp, const unsigned num);

#endif // HELPER_HPP
