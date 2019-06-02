#!/bin/bash
#,2,4,8,16,32,64,128]
forest_sizes=(1 2 4 8 16 32 64 128)

for i in ${forest_sizes[@]};
do
	printf "Generating v1 data for forest of size %i\n" $i
	python v1_results.py $i
done

printf "Done generating v1 data - now generating for v2 data\n\n"

for i in ${forest_sizes[@]};
do
	printf "Generating v2 data for forest of size %i\n" $i
	python v2_results.py $i
done
