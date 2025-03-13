#!/bin/bash

for i in {3..7}
do
    python run_experiments.py --seed $i --settings settings/settings1_linear.yaml > outputs/settings1_linear_seed_$i.out
    python run_experiments.py --seed $i --settings settings/settings1_nonlinear.yaml > outputs/settings1_nonlinear_seed_$i.out
    python run_experiments.py --seed $i --settings settings/settings2_linear.yaml > outputs/settings2_linear_seed_$i.out
    python run_experiments.py --seed $i --settings settings/settings2_nonlinear.yaml > outputs/settings2_nonlinear_seed_$i.out
    python run_experiments.py --seed $i --settings settings/settings3_linear.yaml > outputs/settings3_linear_seed_$i.out
    python run_experiments.py --seed $i --settings settings/settings3_nonlinear.yaml > outputs/settings3_nonlinear_seed_$i.out
done