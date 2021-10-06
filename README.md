# Chirp-Z DFT

Single precision CPU implementation of 1D, 2D and 3D Chirp Z-transforms. The results are verified with FFTW.

## Build

### Prerequisites

The following libraries have to be loaded in noctua in order to build the target

- Intel FPGA SDK for OpenCL, Nallatech BSP (tested with the following versions)

   `module load intelFPGA_pro/21.2.0 nalla_pcie/20.4.0_hpc`

- FFTW (tested with 3.3.8 and 3.3.9):
  
   `module load numlib/FFTW`

- C/C++ compiler (tested with gcc 10.3.0):

    `module load compiler/GCC`

- CMake (tested with 3.20.1)

   `module load devel/CMake`

```bash
module load intelFPGA_pro/21.2.0 nalla_pcie/20.4.0_hpc numlib/FFTW devel/CMake compiler/GCC
```

### Build examples

```bash
# Release Build
cmake -DCMAKE_BUILD_TYPE=Release ..

# Debug build to print a comparison output of fftw and implementation 
cmake -DCMAKE_BUILD_TYPE=Debug ..

# Build the tests and src code
make all

# build only src 
make chirpz

# build only test 
make test_chirpz
```


## Execution

The arguments available to the program:

```bash
Usage:
  Chirp-Z [OPTION...]
  -n, --num arg    Size of FFT dim (default: 31)
  -d, --dim arg    Number of FFT dim (default: 3)
  -p, --path arg   Path to bitstream
  -c, --cpu-only   CPU FFTW Only
  -i, --iter arg   Number of iterations (default: 1)
  -y, --noverify   No verification
  -b, --batch arg  Num of even batches (default: 1)
  -s, --usesvm     SVM enabled
  -h, --help       Print usage
```

To execute:

```bash
# 1D Chirp
./chirpz --cpu-only -n 84 -d 1

# 2D Chirp
./chirpz --cpu-only -n 31 -d 2

# 3D Chirp
./chirpz --cpu-only -n 3 -d 3

# Run all tests
./test_chirpz
```

## Result Interpretation

- When using a `Release` build, the output would denote that it was verified with FFTW and that the output is correct or incorrect. To print the resulting array, use a `DEBUG` build.

```bash
CONFIGURATION: 
---------------
Type        : Complex to Complex
Points      : 31 
Bitstream   : 
Iterations  : 1
Batch       : 1
----------------

-- 1D Chirp
-- Executing ...
-- Verifying ...
-- Works and verified using FFTW
```

or

```bash
...
-- FFTW and Implementation not the same!
```
