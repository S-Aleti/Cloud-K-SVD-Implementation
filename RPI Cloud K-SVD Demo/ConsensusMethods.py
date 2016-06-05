from mpi4py import MPI
from random import randint
import numpy as np
from numpy import linalg as LA
import csv
import time

# Auxillary functions used to collect/share data on a distributed network

def writeWeights(comm,c,degrees):
	# Finds the Metropolis-Hastings weights for a network
	# ========================================================================
	# INPUT ARGUMENTS:
	#   comm                 (MPI.COMM_WORLD)
	#   c                    (MPI.Intracomm.GraphObject) 
	#   degrees 		     (np.array) contains the number of edges that each 
	# 									node's neighbors have
	# ========================================================================
	# OUTPUT: 
	#   weights              (np.array) Metropolis-Hastings weights 
	# ========================================================================

	# Find Metropolis-Hastings weights
	rank = comm.Get_rank()
	size = comm.Get_size()
	weights = np.zeros(size)

	for x in (c.Get_neighbors(rank)):
		weights[x] = 1/(max(degrees[x],degrees[rank])+1)
		
	weights[rank] = 1 - sum(weights[:])

	return weights


def transmitData(z,comm_worldObject,graphObject,node_names,transmissionTag,timeOut):
	# Transmits data with time outs for messaging (only for data_size<=100)k
	# ========================================================================
	# INPUT ARGUMENTS:
	#   z                    (np.array) own data
	#   comm_worldObject     			communicator for MPI4Py
	#   GraphObject 		 			graph object for MPI4py
	#   node_names			 (list(strings)) names of each node 
	#   transmissionTag      (string) 	tag to identify transmission belonging 
    #								  	this algorithm
    #   timeOut 			 (scalar) 	wait time before ending transmission
	# ========================================================================
	# OUTPUT: 
	#   data                 (np.array) data collected from each adjacent node
	# ========================================================================

	# Set up vars
	comm,c = comm_worldObject,graphObject
	size = comm.Get_size()
	rank = comm.Get_rank()
	data_size = np.shape(z)[0] #data_size refers to its dimension
	data = [None]*size

	if data_size<=100:

		# Transmits data repeatedly over the "timeOut" value 
		transmit_data = z
		status = [False]*size
		receiver = [MPI.REQUEST_NULL]*size
		retry = []
		tic = time.time()

		# Sends out data and stores receiver objects
		for j in c.Get_neighbors(rank):
			comm.send(transmit_data,j,tag=transmissionTag)
		for j in c.Get_neighbors(rank):
			rectemp = comm.irecv(source=j,tag=transmissionTag)
			receiver[j] = (rectemp)

		# Receiver objects constantly check for incoming data over timeOut period
		time.sleep(0.050) 						   # waits 50 ms for data by default
		for j in c.Get_neighbors(rank):
			status[j],data[j] = receiver[j].test() # status and data values

		# List of neighbors that data has not been downloaded from yet
		retry = [i for i, node_status in enumerate(status) if (node_status==False)]	

		intialTime = time.time()
		while (retry != []): # while there are receivers needing retries

			for j in retry:

				if (time.time()-intialTime) > timeOut:
					retry.remove(j) #remove from retry list after timeOut
					break
				else: 
					status[j],data[j] = receiver[j].test()
					if status[j]: #status will be rue if data was succesfully recv'd
						retry.remove(j) #remove from retry list after receiving data
			

	else: # breaks up data into pieces for transmitting, necesary for MPI transmissions

		# Recursive function, should only call itself (z_pieces) times in total
		data = [ np.matrix(np.zeros((data_size,1))) ] * size #deals with none assignment
		z_pieces = int(np.ceil(data_size/100))
		working_nodes = c.Get_neighbors(rank)

		for piece in xrange(0,z_pieces): 			# Sends out data for each peice

			data_fragment = z[piece*100:(piece+1)*100]
			node_fragments = transmitData(data_fragment,comm_worldObject,graphObject,
				node_names,transmissionTag,timeOut) # Recursion with fragment

			for j in working_nodes: 				# Only checks 'working' nodes

				if node_fragments[j] is None: 		# If one transmission from a node is messed up
					working_nodes.remove(j)   		# Remove it from the working nodes list
					data[j] = None 			        # 'drop' the data whole
				else: # otherwise just add the node's data fragment
					data[j][piece*100:(piece+1)*100] = node_fragments[j]

	data[rank] = z # its own data
	return data


def correctiveConsensus(z,tc,weights,comm_worldObject,graphObject,node_names,
						transmissionTag,CorrectiveSpacing,timeOut):
	# Performs Corrective Consensus- detailed in the write-up
	# Original paper: (ieeexplore.ieee.org/iel5/5707200/5716927/05717925.pdf)
	# ========================================================================
	# INPUT ARGUMENTS:
	#   z                    (np.array) own data
	#   tc 					 (int) 		total iterations of the algorithm
	#   weights  			 (np.array) Metropolis-Hastings weights
	#   comm_worldObject     			communicator for MPI4Py
	#   GraphObject 		 			graph object for MPI4py
	#   node_names			 (list(strings)) names of each node 
	#   transmissionTag      (string) 	tag to identify transmission belonging 
    #								  	this algorithm
    #   CorrectiveSpacing    (int) 		iterations of regular consensus before a 
    # 							   		corrective iteration
    #   timeOut 			 (scalar) 	wait time before ending transmission
	# ========================================================================
	# OUTPUT: 
	#   qnew                 (np.array) data averaged with network
	# ========================================================================	

	# Set up variables
	datadim = z.shape[0]
	rank = comm_worldObject.Get_rank()
	size = comm_worldObject.Get_size()
	q,qnew = (z),(z)
	phi = np.matrix(np.zeros((datadim,size)))
	CorrCount = 1

	for consensusIteration in xrange(0,tc):

		# Regular Iteration 
		if (consensusIteration != CorrCount*(CorrectiveSpacing+1)-1): 
		    
			q = qnew #set q(t) = q(t-1), remember qnew is from last iteration
			
			# Transfer data
			data = transmitData(qnew,comm_worldObject,graphObject,node_names,
				transmissionTag,timeOut)
			# Consensus
			tempsum = 0 	# used for tracking 'mass' transmitted

			for j in graphObject.Get_neighbors(rank):

				if (data[j] is not None): # if data was received, then...
					difference = data[j]-q 
					tempsum += weights[j]*(difference) #'mass' added to itself
					phi[:,j] += np.reshape((weights[j]*(difference)),(datadim,1))

			#update local data
			qnew = q + tempsum # essentially doing consensus the long way

		# Corrective Iteration
		else:

			# Delta matrix explained in original "Corrective Consensus" paper
			Delta = np.matrix(np.zeros((datadim,size)))
			v = (np.zeros(size))

			# Mass distribution difference is transmitted
			phidata = transmitData(phi[:,j],comm_worldObject,graphObject,node_names,
				transmissionTag,timeOut)

			# Mass distribution should be equivalent sent and received
			for j in graphObject.Get_neighbors(rank):
				if (phidata[j] != None): #if 'mass disr.' transmision was succesful
					Delta[:,j] = phi[:,j] + phidata[j]
					v[j] = 1

			# Ensures stability if packet loss during corrective transmission
			phi[:,j] += (-0.5)*Delta[:,j]*v[j]
			qnew += -(0.5)*np.reshape(np.sum(Delta,1),(datadim,1))

			CorrCount += 1

	return qnew  # q(t) after processing

def powerMethod(M,tc,tp,weights,comm_worldObject,graphObject,node_names,
						transmissionTag,CorrectiveSpacing,timeOut):
	# Performs the Distributed Power Method detailed in the write-up and the 
	# Cloud K-SVD paper
	# ========================================================================
	# INPUT ARGUMENTS:
	#   M 					 (np.array) obtained from SVD decomp of error 
	# 								    matrix, explained in paper 
	#   tc 					 (int) 		iterations of corrective consensus
	#   tp  			     (int) 		iterations of the power method
	#   weights  			 (np.array) Metropolis-Hastings weights
	#   comm_worldObject     			communicator for MPI4Py
	#   GraphObject 		 			graph object for MPI4py
	#   node_names			 (list(strings)) names of each node 
	#   transmissionTag      (string) 	tag to identify transmission belonging
    #								  	this algorithm
    #   CorrectiveSpacing    (int) 		iterations of regular consensus before 
    # 							   		a corrective iteration
    #   timeOut 			 (scalar) 	wait time before ending transmission
	# ========================================================================
	# OUTPUT: 
	#   eigenvector          (np.array) new k^th atom, detailed in paper
	# ========================================================================	

    # Set up vars
    datadim = M.shape[0]
    rank = comm_worldObject.Get_rank()
    size = comm_worldObject.Get_size()
    q = np.matrix(np.ones((datadim,1)))
    qnew = q
    phi = np.matrix(np.zeros((datadim,size)))
    CorrCount = 1

 	# Run the distributed power method
    for powerIteration in range(0,tp,1): # each iteration of the power method

        qnew = (M*qnew) # We use corrective consensus here, regular consensus works too
        qnew = correctiveConsensus(qnew,tc,weights,comm_worldObject,graphObject,node_names,
						transmissionTag,CorrectiveSpacing,timeOut)

        if LA.norm(qnew) != 0:
        	qnew /= LA.norm(qnew) # normalize

    eigenvector = qnew.reshape(datadim) 

    return eigenvector 
    
