#!/bin/bash

forest_sizes=(1 2 4 8 16 32 64 128)

for i in ${forest_sizes[@]};
do
	python v1_show_data.py $i 
	python v2_show_data.py $i 
done
