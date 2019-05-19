#!/bin/bash
#,2,4,8,16,32,64,128]
forest_sizes=(1 2 4 8 16 32 64 128)

for i in ${forest_sizes[@]};
do
	printf "Generating models for covtype with %i trees\n" $i
	python gen_forest.py $i '1'
done


for i in ${forest_sizes[@]};
do
	printf "Generating models for susy-type with %i trees\n" $i
	python gen_forest.py $i '0'
done
