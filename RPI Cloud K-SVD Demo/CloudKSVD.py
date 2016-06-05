import numpy as np
import scipy as sp
from numpy import linalg as LA
from mpi4py import MPI
from MainMethods import *
from MNIST_Loader import *
from ConsensusMethods import *
import sys

## Network Setup
comm = MPI.COMM_WORLD
comm1= MPI.Intracomm(comm)
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()
name = MPI.Get_processor_name()
index = [3,6,9,12]
edges = [1,2,3,0,2,3,0,1,3,0,1,2] #						Sparsely connected network of 4 nodes
c = comm1.Create_graph(index, edges, reorder = False) 
np.set_printoptions(precision=3)
tic = time.time()


## Prelims

node_names = list('ABCDE')
degrees = discoverDegrees(c,comm,node_names)
weights = writeWeights(comm,c,degrees)


## Signals

folder = 'python-mnist/data'
resolution = (24,24)
ddim = resolution[0]*resolution[1]  #Data dimension
classes = [0,3,5,8,9]				#MNIST digits to collect
amount = 10  #Samples of atoms for each class; K = len(classes)*amount
signals = 20 #Samples in Y


## Collect MNIST data

time.sleep(0.1)
tic = time.time()
D,D_labels,Y,Y_labels = importMNIST(folder,resolution,classes,amount,signals)
S = signals				# Same as in paper
K = np.shape(D)[1] 		# Same as in paper 
ddim = np.shape(D)[0]
print 'done'


## Cloud params (some may be command line args)

try:     # check for command line args

	args = sys.argv
	tD = int(args[1]) 	# cloud kSVD iterations
	t0 = int(args[2]) 	# sparsity
	tc = int(args[3])	# consensus iterations
	tp = int(args[4])	# power iterations

except IndexError:

	print 'Using default parameters because error: incorrect number of arguments detected'
	print 'Usage: mpiexec -n <N> --hostifle <./hostfile> python CloudkSVD.py <tD> <t0> <tc> <tp>'
	tD,t0,tc,tp = 5,5,2,2

except ValueError:

	print 'Using default parameters because error: arguments must be integers'
	tD,t0,tc,tp = 5,5,2,2

print tD, t0, tc, tp, args


## Network params

refvec = np.matrix(np.ones((ddim,1))) # Q_init for power method, sets direction of result
Tag = 11			  			      # Transmission tag, ensures MPI transmissions don't interfere
CorrectiveSpacing = 3 			      # Regular iterations before a corrective iteration
timeOut = 0.150		  				  # Time nodes wait before they move on


## Main

time_sync(tic,15) # Nodes must start at approximately the same time
				  # else segmentation fault or massive packet loss
print 'starting C-KSVD'

rt0 = time.time()
D,X,rerror = CloudKSVD(D,Y,refvec,tD,t0,tc,tp,weights,comm,c,node_names,
	Tag,CorrectiveSpacing,timeOut)
rt1 = time.time()
rt = rt1 - rt0	# total run time of cloud kSVD

## Data Collection
f = open('running_time_log.txt','a+')
buf1 = '\nResolution: ' + str(resolution) + '\nClasses: ' + str(classes) + '\nAmount per class: %d \nSignals: %d \n' % (amount, signals)
buf2 = '\ntD: %d \nt0: %d \ntc: %d \ntp: %d \n' % (tD,t0,tc,tp)
buf3 = '\nTotal ckSVD running time: %9.6f \n' % rt
f.write(buf1 + buf2 + buf3)
f.close()


## Results

error = np.linalg.norm(Y-np.dot(D,X))**2 # L2 norm squared for error

# Residuals
c_atoms = []

for c in classes:
	c_atoms.append(list([i for i,x in enumerate(D_labels) if x == c ])) # atoms corresponding to each class

residual = np.matrix(np.zeros((S,len(classes))))						# residuals

for s in xrange(0,S):				  # for each signal
	for c in xrange(0,len(classes)):  # for each class
		if len(c_atoms[c])>0:
			residual[s,c] = LA.norm( Y[:,s] - D[:,c_atoms[c]]*X[c_atoms[c],:] )
		else:
			residual[s,c] = 1000 # magic num residual if no relevant atoms detected, likely will never see this in practice


guesses = [classes[mins] for mins in list([np.argmin(residual[s,:]) for s in list(xrange(0,S))])] # predicted classes
actual  = list(Y_labels)																	      # actual classes

if rank == 0:
	print residual
	print guesses
	print actual

accuracy = float(sum(np.array(guesses)==np.array(actual)))/S  # classification accuracy

print "Node %s accuracy: %d%s" % (node_names[rank], accuracy*100,"%")
