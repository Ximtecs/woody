#!/bin/bash
#,2,4,8,16,32,64,128]
forest_sizes=(1 2 4 8 16 32 64 128)
print_cpu='0'

for i in ${forest_sizes[@]};
do
	python gen_plot.py $i print_cpu
	python gen_susy_plot.py $i print_cpu
done
