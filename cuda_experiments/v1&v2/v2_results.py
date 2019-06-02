import time
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from woody import WoodClassifier
from woody.data import *
import sys

seed = 0
covtype_size = [100000, 150000, 200000, 250000, 300000, 350000, 400000]
Xtrain, ytrain, Xtest, ytest = covtype(train_size=400000, seed=seed)
print(len(Xtrain))

new_arr = np.repeat(Xtrain, 10,axis=0)

print(len(new_arr))
total_time = []

if (len(sys.argv) != 2 ):
	print("Specify number of trees in forest (1,2,4,8,16...)")
else:
	print("Generating data for %s trees" % (sys.argv[1]))

nr_tree = int(sys.argv[1])

#file_name = 'model_' + str(nr_tree) + 'tree.data'
file_name = './models/model_' + str(nr_tree) + 'tree_4jobs.data'
model = WoodClassifier.load(file_name)
file_name = './models/model_' + str(nr_tree) + 'tree.data'
model2 = WoodClassifier.load(file_name)
nr_classes = len(np.unique(ytrain)) +1 #not sure if accurate
model.compile_store_v2(new_arr,nr_classes,10)

print("Number of estimators: \t\t%i" % model.n_estimators)
forest_time = []

for i in range(len(covtype_size)-1,-1,-1):
	times = np.zeros(5,np.float32)
	X_temp = new_arr[:covtype_size[i]]
	print("Number of training patterns:\t%i" % X_temp.shape[0])
	start_time = time.time()
	cpu_test = model.predict(X_temp)
	cpu_test = model.predict(X_temp)
	cpu_test = model.predict(X_temp)
	end_time = time.time()
	print("Total time for 3*predict 4jobs %f"% (end_time - start_time))
	times[0] = (end_time - start_time) / 3.0

	start_time = time.time()
	cpu_test = model2.predict(X_temp)
	cpu_test = model2.predict(X_temp)
	cpu_test = model2.predict(X_temp)
	end_time = time.time()
	print("Total time for 3*predict 1jobs %f"% (end_time - start_time))
	times[4] = (end_time - start_time) / 3.0


	start_time = time.time()
	v2, t2, q2, m2, tb2, c2 = model.cuda_v2(X_temp,10,0)
	v2, t2, q2, m2, tb2, c2 = model.cuda_v2(X_temp,10,0)
	v2, t2, q2, m2, tb2, c2 = model.cuda_v2(X_temp,10,0)
	end_time = time.time()
	print("Total time for 3*cuda_base %f"% (end_time - start_time))
	times[1] = (end_time - start_time) / 3.0

	start_time = time.time()
	v3, t3, q3, m3, tb3, c3 = model.cuda_v2(X_temp,10,1)
	v3, t3, q3, m3, tb3, c3 = model.cuda_v2(X_temp,10,1)
	v3, t3, q3, m3, tb3, c3 = model.cuda_v2(X_temp,10,1)
	end_time = time.time()
	print("Total time for 3*cuda_mult %f"% (end_time - start_time))
	times[2] = (end_time - start_time) / 3.0

	start_time = time.time()
	v4, t4, q4, m4, tb4, c4 = model.cuda_v2(X_temp,10,2)
	v4, t4, q4, m4, tb4, c4 = model.cuda_v2(X_temp,10,2)
	v4, t4, q4, m4, tb4, c4 = model.cuda_v2(X_temp,10,2)
	end_time = time.time()
	print("Total time for 3*cuda_for %f"% (end_time - start_time))
	times[3] = (end_time - start_time) / 3.0



	forest_time.append(times)


forest_time = forest_time[::-1]

res_file_name = './results/v2_' + str(nr_tree) + 'tree_results.p'
pickle.dump(forest_time,open(res_file_name,'wb'))