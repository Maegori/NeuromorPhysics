#!/usr/bin/env bash

#zdims: 5, 8, 13, 21, 24, 55, 89, 144
#data_split:0.1

#run results.py with all zdims and data_split
for spi in 10 15 20 25 30 35 40 45 50 100
do
    for zdim in 5 8 13 21 24 55 89 144
    do
        for data_split in 0.1
        do
            /homes/lexjohan/nengoloihi/miniconda/bin/python results.py --zdim $zdim --data_split $data_split --steps_per_sample $spi
        done
    done
done