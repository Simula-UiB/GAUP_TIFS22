#!/bin/bash

export OMP_NUM_THREADS=4
echo "Number of threads: $OMP_NUM_THREADS (applicable only if compiled with OpenMP)"

for N in {1..4}
do
  echo "Runnin for N=$N"
  time ./gauss_dataset_quant 3 $N gauss_data_csv/gaussTrain1M_N$N.csv gauss_data_csv/gaussTest100k_N$N.csv 10 > output/N$N.log
done
