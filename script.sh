#!/bin/bash


for T in $(seq 20 -1 1)
do
    echo "Running Fed_Unlearn_main.py with T_epoch=$T"
    python Fed_Unlearn_main.py --T_epoch $T
done
