import numpy as np 
import scipy as sp
import time
from numpy import linalg as LA
from ConsensusMethods import *
from mpi4py import MPI

# Primary functions used to run Cloud K-SVD

def time_sync(tic,wait_period): 
    # Makes sure a node waits until (wait_period) is over.
    # Simpler to use MPI barrier command but that defeats the purpose of a 
    # distributed network. 
    # ========================================================================
    # INPUT ARGUMENTS:
    #   tic                  (float) time when algorithm was called 
    #   wait_period          (float) time to wait
    # ========================================================================

    current = time.time()
    while current<(tic+wait_period): 
        # print 'waiting...'
        time.sleep(0.1)
        current = time.time()  


def discoverDegrees(c,comm,node_names):
    # Figures out the degree matrix of the network by contacting neighbors.
    # Necesary to know degrees of neighbors for Metropolis-Hastings Weights
    # ========================================================================
    # INPUT ARGUMENTS:
    #   c                    (MPI.Intracomm.GraphObject) 
    #   comm                 (MPI.COMM_WORLD)
    #   node_names           (list(strings)) name of each node
    # ========================================================================
    # OUTPUT: 
    #   degrees              (np.array) contains the number of edges that each 
    #                                   node's neighbors have
    # ========================================================================

    # Find size of network and number of neighbors
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    print "I am %s, my degree is: %d" % (node_names[rank], 
        c.Get_neighbors_count(rank)) , " and my neighbors are: ", c.Get_neighbors(rank)

    # Create degree matrix
    degrees = np.zeros(size)
    tempedges = (c.Get_neighbors(rank))
    tempedges.append(rank)

    for x in (tempedges):
        comm.send(c.Get_neighbors_count(rank)+1,x, tag=7)
        degrees[x] = comm.recv(source=x, tag=7)

    print "I am %s, my degree matrix is: " % (node_names[rank]) , degrees

    return degrees


def OMP(D,Y,L):
    # Runs the OMP algorithm
    # Very rarely has a convergence issue with pinv function
    # ========================================================================
    # INPUT ARGUMENTS:
    #   D                    (np.array) dictionary 
    #   Y                    (np.array) signals 
    #   L                    (int)      sparsity constriant 
    # ========================================================================
    # OUTPUT: 
    #   A                    (np.array) coefficient matrix used to reproduce
    #                                   the signals: D*A ~= Y
    # ========================================================================

    # Same variable names as paper
    N = D.shape[0]
    K = D.shape[1]
    P = Y.shape[1]
    A = np.matrix('')

    # Ensure dimension of dictionary is same as that of signals
    if(N != Y.shape[0]):
        print "Feature-size does not match!"
        return

    for k in range(0,P):

        a = []
        x = Y[:,k]
        residual = x
        indx = [0]*L

        for j in range(0,L):
            proj = np.dot(np.transpose(D),residual)
            k_hat = np.argmax(np.absolute(proj))
            indx[j] = k_hat
            t1 = D[:,indx[0:j+1]]
            a = np.dot(np.linalg.pinv(t1),x)
            residual = x - np.dot(D[:,indx[0:j+1]],a) 
            if(np.sum(np.square(residual)) < 1e-6):    #1e-6 = magic number to quit pursuit
                break
        temp = np.zeros((K,1))
        temp[indx[0:j+1]] = a;

        if (A.size == 0):
            A = temp
        else:
            A = np.column_stack((A,temp))

    return A


def CloudKSVD(D,Y,refvec,tD,t0,tc,tp,weights,comm,c,node_names,Tag,
    CorrectiveSpacing,timeOut): 
    # Runs the Cloud K-SVD algorithm on a given network
    # ========================================================================
    # INPUT ARGUMENTS:
    #   D                    (np.array) dictionary 
    #   Y                    (np.array) signals 
    #   refvec               (np.array) reference vector to ensure all vectors
    #                                   found by the nodes point the same way
    #   tD                   (int)      iterations of Cloud K-SVD
    #   t0                   (int)      sparsity constraint
    #   tc                   (int)      total iterations of corrective consensus
    #   tp                   (int)      total iterations of the power method
    #   weights              (np.array) Metropolis-Hastings weights
    #   comm                 (MPI.COMM_World)
    #   c                    (graphObject) 
    #   node_names           (list(strings)) names of each node 
    #   tag                  (string)   tag to identify transmission belonging 
    #                                   this algorithm
    #   CorrectiveSpacing    (int)      iterations of regular consensus before a 
    #                                   corrective iteration
    #   timeOut              (scalar)   wait time before ending transmission
    # ========================================================================
    # OUTPUT: 
    #   D                    (np.array) approximate cloud dictionary
    #   x                    (np.array) coefficient matrix used to reproduce   
    #                                   local signals
    # ========================================================================  

    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    ddim = np.shape(D)[0]
    K = np.shape(D)[1]
    S = np.shape(Y)[1]
    x = np.matrix(np.zeros((K,S)))
    rerror = np.zeros(tD)

    for t in xrange(0,tD): #iterations of kSVD

        if rank == 0:
            print '=================Iteration %d=================' % (t+1)

        for s in xrange(0,S):
            x[:,s] = OMP(D,Y[:,s],t0)

        for k in xrange(0,K):
            if rank == 0:
                print 'Updating atom %d' % (k+1)
            #Error matrix
            wk = [i for i,a in enumerate((np.array(x[k,:])).ravel()) if a!=0]
            Ek = (Y-np.dot(D,x)) + (D[:,k]*x[k,:])
            ERk = Ek[:,wk]

            #Power Method
            if np.size(wk) == 0: #if empty
                M = np.matrix(np.zeros((ddim,ddim)))
            else:
                M = ERk*ERk.transpose()
            q = powerMethod(M,tc,tp,weights,comm,c,node_names,
                            Tag,CorrectiveSpacing,timeOut)

            #Codebook Update
            if np.size(wk) != 0: #if not empty
                refdirection = np.sign(np.array(q*refvec)[0][0])
                if LA.norm(q) != 0:
                    D[:,k] = (refdirection*(q/(LA.norm(q)))).reshape(ddim,1)
                else:
                    D[:,k] = q.reshape(ddim,1)
                x[k,:] = 0
                x[k,wk]= np.array(D[:,k].transpose()*ERk).ravel()

        #Error Data
        rerror[t] =np.linalg.norm(Y-np.dot(D,x))
        print "Node %s Iteration %d error:" % (node_names[rank],t+1) , rerror[t]
        time.sleep(0.2)

    return D,x,rerror


def ActiveDictionaryFilter(D,Y,NewSignals,T0): 
    # Based on "Active dictionary learning for image representation" - Wu, 
    # Sawarte, & Bajwa. When a new signal is added to Y, rather than calling 
    # K-SVD or Cloud K-SVD on all the signals, the function simply returns a 
    # new matrix with the "worst" represented signal added on. After calling 
    # this, you can call either Cloud or Local K-SVD with the new training 
    # pool P that includes the worst represented signal
    # ========================================================================
    # INPUT ARGUMENTS:
    #   D                    (np.array) dictionary 
    #   Y                    (np.array) signals 
    #   NewSignal            (np.array) new signals to append to the old ones
    #   T0                   (int)      sparsity constraint
    # ========================================================================
    # OUTPUT: 
    #   P                    (np.array) signals with the largest reproduction 
    #                                   error (L2 norm)
    #   error                (np.array) largest L2-norm error of the signals
    # ========================================================================  

    # Variables set to same names as paper
    Yt = NewSignals
    N  = np.shape(Yt)[1]        # Number of new signals
    Theta = OMP(D,Yt[:,i],T0)
    Yhat = np.matrix(D*Theta)   # Sparse representation of new signals with given dictionary
    rep_error = np.matrix(np.zeros(N))

    for i in xrange(1,N):       # Determines worst represented signal
        rep_error[i] = LA.norm(Y[:,i]-Yhat[:,i])**2 #L2 Norm Squared

    S = Yt[:,np.argmax(rep_error)] # Pick the worst represented signal
    P = np.concatenate((P,S),1)    # Add it to the training pool
    error = np.argmax(rep_error)

    return P,error  # Returns training pool and index of signal from batch




