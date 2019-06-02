from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys
if (len(sys.argv) != 2 ):
	print("Specify number of trees in forest (1,2,4,8,16...)")
else:
	print("Plotting data for %s trees" % (sys.argv[1]))

nr_tree = int(sys.argv[1])
file_name = './results/v1_' + str(nr_tree) + 'tree_results.p'
results = np.array(pickle.load(open(file_name,'rb')))

covtype_size = [100000, 150000, 200000, 250000, 300000, 350000, 400000]
covtype_size = [float(i) / 100000. for i in covtype_size]
plt.figure(1)
plt.plot(covtype_size,results[:,8],label='CPU',linewidth=5.0,color='k',linestyle="-")
plt.plot(covtype_size,results[:,0],label='cuda_tree',linewidth=5.0,linestyle=":",color='g')
plt.plot(covtype_size,results[:,1],label='cuda_tree_mul',linewidth=5.0,linestyle=":",color='y')
plt.plot(covtype_size,results[:,2],label='cuda_forest',linestyle="--",color='b',linewidth=5.0)
plt.plot(covtype_size,results[:,3],label='cuda_forest_mul',linestyle=":",linewidth=5.0)
plt.plot(covtype_size,results[:,4],label='cuda_all',linestyle="-.",color='r',linewidth=5.0)
plt.plot(covtype_size,results[:,5],label='cuda_all_mul',linestyle=":",linewidth=5.0)
if(nr_tree < 64):
	plt.plot(covtype_size,results[:,6],label='cuda_block',linewidth=5.0,color='m', linestyle=':')
	plt.plot(covtype_size,results[:,7],label='cuda_block_mul',linewidth=5.0,color='m',linestyle=':')
plt.legend(loc='best')

plt.ylabel('time [ s ]',fontsize=20)
plt.xlabel('test instances [10^5]',fontsize=20)
title = 'Covtype v1, ' + str(nr_tree) + ' tree(s) '
plt.title(title,fontsize=20)

pic_file_name = "./figs/v1_" + str(nr_tree) + 'tree.png'
plt.savefig(pic_file_name)