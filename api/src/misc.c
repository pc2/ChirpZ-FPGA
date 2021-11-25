// Author: Arjun Ramaswami

#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#define _USE_MATH_DEFINES

#include "misc.h"
/**
 * \brief  compute walltime in milliseconds
 * \return time in milliseconds
 */
double getTimeinMilliSec(){
   struct timespec a;
   if(clock_gettime(CLOCK_MONOTONIC, &a) != 0){
     fprintf(stderr, "Error in getting wall clock time \n");
     exit(EXIT_FAILURE);
   }
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

unsigned next_power_of_two(unsigned x){
  x = x - 1; 
  x = x | (x >> 1); 
  x = x | (x >> 2); 
  x = x | (x >> 4); 
  x = x | (x >> 8); 
  x = x | (x >> 16); 
  x = x + 1;
  return x;
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