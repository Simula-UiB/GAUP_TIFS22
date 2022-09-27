#!/bin/bash

export OMP_NUM_THREADS=4
echo "Number of threads: $OMP_NUM_THREADS"

for N in {1..4}
do
  echo "Runnin for N=$N"
  time ./gauss_dataset_quant 3 $N gaussTrain1M_N$N.csv gaussTest100k_N$N.csv 10 > N$N.log
done
