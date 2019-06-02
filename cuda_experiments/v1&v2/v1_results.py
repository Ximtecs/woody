import time
import numpy as np
from sklearn.metrics import accuracy_score
import pickle
from woody import WoodClassifier
from woody.data import *
import sys

seed = 0
covtype_size = [100000, 150000, 200000, 250000, 300000, 350000, 400000]
forest_size = [1, 2, 4, 8, 16, 32, 64, 128]

Xtrain_global, ytrain, Xtest, ytest = covtype(train_size=400000, seed=seed)
total_time = []

if (len(sys.argv) != 2 ):
	print("Specify number of trees in forest (1,2,4,8,16...)")
else:
	print("Generating data for %s trees" % (sys.argv[1]))

nr_tree = int(sys.argv[1])


#file_name = 'model_' + str(nr_tree) + 'tree.data'import sys
file_name = './models/model_' + str(nr_tree) + 'tree_4jobs.data'
model = WoodClassifier.load(file_name)
nr_classes = len(np.unique(ytrain)) +1 #not sure if accurate
model.compile_and_Store(Xtrain_global,nr_classes)
print("Number of estimators: \t\t%i" % model.n_estimators)
forest_time = []

for i in xrange(len(covtype_size)):
	times = np.zeros(9,np.float32)
	Xtrain = Xtrain_global[:covtype_size[i]]
	print("Number of training patterns:\t%i" % Xtrain.shape[0])
	start_time = time.time()
	v1 = model.cuda_predict(Xtrain)
	v1 = model.cuda_predict(Xtrain)
	v1 = model.cuda_predict(Xtrain)
	end_time = time.time()
	times[0] = (end_time - start_time) / 3.0
	
	start_time = time.time()
	v2 = model.cuda_pred_tree_mult(Xtrain,10)
	v2 = model.cuda_pred_tree_mult(Xtrain,10)
	v2 = model.cuda_pred_tree_mult(Xtrain,10)
	end_time = time.time()
	times[1] = (end_time - start_time) / 3.0


	start_time = time.time()
	v3 = model.cuda_pred_forest(Xtrain)
	v3 = model.cuda_pred_forest(Xtrain)
	v3 = model.cuda_pred_forest(Xtrain)
	end_time = time.time()
	times[2] = (end_time - start_time) / 3.0	


	start_time = time.time()
	v4 = model.cuda_pred_forest_mult(Xtrain,10)
	v4 = model.cuda_pred_forest_mult(Xtrain,10)
	v4 = model.cuda_pred_forest_mult(Xtrain,10)
	end_time = time.time()
	times[3] = (end_time - start_time) / 3.0		


	start_time = time.time()
	v5 = model.cuda_pred_all(Xtest)
	v5 = model.cuda_pred_all(Xtest)
	v5 = model.cuda_pred_all(Xtest)
	end_time = time.time()
	times[4] = (end_time - start_time) / 3.0

	start_time = time.time()
	v6 = model.cuda_pred_all_mult(Xtrain,10)
	v6 = model.cuda_pred_all_mult(Xtrain,10)
	v6 = model.cuda_pred_all_mult(Xtrain,10)
	end_time = time.time()
	times[5] = (end_time - start_time) / 3.0


	start_time = time.time()
	if (nr_tree < 64):
		v7 = model.cuda_pred_1block1X(Xtrain)
		v7 = model.cuda_pred_1block1X(Xtrain)
		v7 = model.cuda_pred_1block1X(Xtrain)
	end_time = time.time()
	times[6] = (end_time - start_time) / 3.0

	start_time = time.time()
	if (nr_tree < 64):
		v8 = model.cuda_pred_1block_multX(Xtrain)
		v8 = model.cuda_pred_1block_multX(Xtrain)
		v8 = model.cuda_pred_1block_multX(Xtrain)
	end_time = time.time()
	times[7] = (end_time - start_time) / 3.0

	start_time = time.time()
	cpu_test = model.predict(Xtrain)
	cpu_test = model.predict(Xtrain)
	cpu_test = model.predict(Xtrain)
	end_time = time.time()
	times[8] = (end_time - start_time) / 3.0

	forest_time.append(times)


res_file_name = './results/v1_' + str(nr_tree) + 'tree_results.p'
pickle.dump(forest_time,open(res_file_name,'wb'))