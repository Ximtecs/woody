from matplotlib import pyplot as plt
import pickle
import numpy as np
import sys


if (not(len(sys.argv) != 2 or len(sys.argv) != 3)):
	print("Specify number of trees in forest (1,2,4,8,16...)")
	print("second argument: 0 or 1 - prints cpu result if set to 1")
else:
	print("Plotting")

nr_tree =sys.argv[1]
file_name = './results/v2_forest_susy' + str(nr_tree) + 'tree_results.p'


file_name = './results/v2_forest_' + str(nr_tree) + 'tree_results.p'


results = np.array(pickle.load(open(file_name,'rb')))
sizes = [100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000]
sizes = [float(i) / 100000. for i in sizes]


plt.figure(1)
plt.plot(sizes,results[:,0],label='Total time',linewidth=5.0,color='k')
plt.plot(sizes,results[:,1],label='Transfer time',linestyle="--",color='k',linewidth=5.0)
plt.plot(sizes,results[:,2],label='Query time',linestyle="-.",color='k',linewidth=5.0)
plt.plot(sizes,results[:,3],label='Vote time',linestyle="-",color='r',linewidth=5.0)
plt.plot(sizes,results[:,4],label='Transfer back time',linestyle=":",color='r',linewidth=5.0)
plt.plot(sizes,results[:,5],label='Cleanup time',linestyle="-.",color='r',linewidth=5.0)
plt.plot(sizes,results[:,6],label='Sum time',linestyle=":",color='k',linewidth=5.0)
if (sys.argv[2] == 1):
	plt.plot(sizes,results[:,7],label='CPU',linewidth=5.0,color='b') #Comment out if you only want to see GPU time
plt.legend(loc='best')

print("CPU final time: %f" % (results[:,7][len(sizes)-1]))
print("GPU final time: %f" % (results[:,0][len(sizes)-1]))
print("speedup: %f" % (results[:,7][len(sizes)-1] / results[:,0][len(sizes)-1]))

plt.ylabel('time [ s ]',fontsize=20)
plt.xlabel('test instances [ 10^5 ]',fontsize=20)
title = 'Covtype, ' + str(nr_tree) + ' tree(s) '
plt.title(title,fontsize=20)

pic_file_name = "./figs/forest_" + str(nr_tree)+ 'tree_v2.png'
plt.savefig(pic_file_name)
#plt.show()

