#!/usr/bin/env bash

for sparsity in 0.3
do

echo $sparsity
python2 prepare_dense_data.py $sparsity

done
