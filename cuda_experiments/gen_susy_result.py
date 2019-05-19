import time
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from woody import WoodClassifier
from woody.data import *
import sys

if (len(sys.argv) != 2 ):
	print("Specify number of trees in forest (1,2,4,8,16...)")

nr_tree = int(sys.argv[1])


seed = 0
Xtrain, ytrain, Xtest, ytest = susy(train_size=4000000, seed=seed)
susy_size = [100000, 500000, 1000000, 2000000, 3000000, 4000000]


new_arr = np.repeat(Xtrain, 1,axis=0) #originally only used for covtype

print(len(new_arr))
total_time = []

file_name = './models/model_susy' + str(nr_tree) + 'tree_4jobs.data'
model = WoodClassifier.load(file_name)
nr_classes = len(np.unique(ytrain)) +1
model.compile_store_v2(new_arr,nr_classes,10)

print("Number of estimators: \t\t%i" % model.n_estimators)
forest_time = []

for i in xrange(len(susy_size)):
	times = np.zeros(8,np.float32)
	X_temp = new_arr[:susy_size[i]]
	print("Number of training patterns:\t%i" % X_temp.shape[0])
	
	start_time = time.time()
	cpu_test = model.predict(X_temp)
	cpu_test = model.predict(X_temp)
	cpu_test = model.predict(X_temp)
	end_time = time.time()
	print("Total time for 3*predict 4jobs %f"% (end_time - start_time))
	times[7] = (end_time - start_time) / 3.0

	tot_time = 0
	transfer_time = 0
	query_time = 0
	vote_time = 0
	transfer_back_time = 0
	clean_up_time = 0
	tot_sum = 0
	for j in xrange(3):
		start_time = time.time()
		v, t, q, m, tb, c = model.cuda_v2(X_temp,10,2)
		end_time = time.time()

		tot_time += end_time - start_time
		transfer_time += t
		query_time += q
		vote_time += m
		transfer_back_time += tb
		clean_up_time += c
		tot_sum = transfer_time + query_time + vote_time + transfer_back_time + clean_up_time

	times[0] = tot_time / 3.0 
	times[1] = transfer_time / 3.0
	times[2] = query_time / 3.0
	times[3] = vote_time / 3.0
	times[4] = transfer_back_time / 3.0
	times[5] = clean_up_time / 3.0
	times[6] = tot_sum / 3.0
	forest_time.append(times)


res_file_name = './results/v2_forest_susy' + str(nr_tree) + 'tree_results.p'
pickle.dump(forest_time,open(res_file_name,'wb'))