#!/bin/bash
#,2,4,8,16,32,64,128]
forest_sizes=(1 2 4 8 16 32 64 128)

for i in ${forest_sizes[@]};
do
	printf "Generating data for forest of size %i\n" $i
	python gen_susy_result.py $i
done

printf "Done generating susy data - now generating for covtype data\n\n"

for i in ${forest_sizes[@]};
do
	printf "Generating data for forest of size %i\n" $i
	python gen_result.py $i
done
