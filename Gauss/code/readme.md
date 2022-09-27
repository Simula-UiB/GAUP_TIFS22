Calculations on synthetic Gaussian data

# Compression-based
The compression-based approach is implemented in two files:
* `gaussLBG.cpp` - the main file
* `utils.cpp` - some utility functions used in the main file

## Compilation
The code can be compiled with using OpenMP or without it. The recipes below are for Linux (tested on Ubuntu Linux), but compilation on other platforms should be similar. Some general notes:
* the C++ code is written with C++17 standard in mind
* use `-O3` (or similar) optimisation level for faster run
* eventually, make use of multiple cores by compiling with support of OpenMP

### With OpenMP
In order to compile with `g++` compiler on Linux for running with OpenMP (make use of more cores), make sure the following line is **not** commented out in the file `gaussLBP.cpp`:
```
#define USE_OMP
```
After that, use the following in terminal:
```
g++ -std=c++17 -O3 -march=native -fopenmp gaussLBG.cpp -o gauss_dataset_quant
```
The executable file will be `gauss_dataset_quant`

### Without OpenMP
In order to compile with `g++` compiler on Linux for running **without** OpenMP (make use of one core only), make sure the following line is commented out in the file `gaussLBP.cpp`:
```
#define USE_OMP
```
After that, use the following in terminal:
```
g++ -std=c++17 -O3 -march=native  gaussLBG.cpp -o gauss_dataset_quant
```
The executable file will be `gauss_dataset_quant`

## Execution
After the file is compiled, it can be run for all the cases by running the file `run_all_compressionbased.sh`. In the bash file, you can change the following line
```
export OMP_NUM_THREADS=4
```
up to the number of cores you would like to use for the computation. It doesn't make sense to make it larger than the number of cores you have in your system. 
Also, perhaps, leave one or two cores for the rest of the system to run.

The results will be output to the files `N1.log`..`N4.log`
