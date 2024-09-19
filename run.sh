!/usr/bin/env bash

# 7200s -> 2h
python main.py --config configs/acc_config.yaml --runtime 7200 --min_budget 5 --max_budget 20 --total_budget 300 --seed 2 &> log2.txt
python main.py --config configs/acc_config.yaml --runtime 7200 --min_budget 5 --max_budget 20 --total_budget 300 --seed 3 &> log3.txt
python main.py --config configs/acc_config.yaml --runtime 7200 --min_budget 5 --max_budget 20 --total_budget 300 --seed 4 &> log4.txt
python main.py --config configs/acc_config.yaml --runtime 7200 --min_budget 5 --max_budget 20 --total_budget 300 --seed 5 &> log5.txt
python main.py --config configs/acc_config.yaml --runtime 7200 --min_budget 5 --max_budget 20 --total_budget 300 --seed 6 &> log6.txt
