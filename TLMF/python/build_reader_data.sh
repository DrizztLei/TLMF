#!/usr/bin/env bash

for sparsity in 0.1 0.2 0.3 0.6 0.7
do

echo $sparsity
python2 prepare_reader_data.py $sparsity

done
