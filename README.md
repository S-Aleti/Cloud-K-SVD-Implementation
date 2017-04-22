# MATLAB Implementation

## [MNIST](http://yann.lecun.com/exdb/mnist/) Simulation

The code in the MNIST Simulation folder can be run for a Monte Carlo Simulation of local [K-SVD](http://www.cs.technion.ac.il/~freddy/papers/120.pdf) and [Cloud K-SVD](http://arxiv.org/abs/1412.7839) using MNIST Data. The parameters for it can be set in the file `MNIST_Analysis.m`, which may then be run for the simulation itself. The code outputs the average representation error and classification accuracy for both forms of K-SVD using the same data. The node networks for Cloud K-SVD are generated randomly each time the code is run, and there is no simulated packet loss. Therefore, the results represent a theoretical best case for the results of Cloud K-SVD. Our simulated results, which were the averages of 100 Monte Carlo trials, are given in the Report pdf. 

## [Corrective Consensus](http://vision.jhu.edu/assets/consensus-cdc10.pdf) Simulation

The Corrective Consensus folder contains a MATLAB simulation of a network perfoming corrective consensus with a variable amount of packet loss. This code may be run with the following command: `ConsensusRobustv2(n,dim,packetloss,iterations,k)` where *n* is the number of nodes, *dim* is the dimension of the data, *packetloss* is the mean packets dropped per iteration, *iterations* is the number of consensus iterations, and k is the number of standard iterations before a corrective iteration. The packet loss is simulated using a poisson random variable, and the network is randomly generated each time the code is run. The output of the code shows each node's data converging to the true average (denoted by the blue-dashed line) over time. A more detailed explanation is given in the discussion of our report.

# RPI Implementation

## MPI Cloud K-SVD Demo

The code to run Cloud K-SVD, Corrective Consensus, and Active Dictionary Learning is provided in this folder. This implementation uses [**OpenMPI**](https://www.open-mpi.org/), which requires multiple dependencies that can be installed with the following commands:

- OpenMPI: `sudo apt-get install openmpi-bin openmpi-common openssh-client openssh-server  libopenmpi-dev`
- Python-Dev: `sudo apt-get install python-dev`
- MPI4Py: `sudo pip install mpi4py`
- Python-MNIST `sudo pip install python-mnist`

The documentation for MPI4Py is available [here](http://pythonhosted.org/mpi4py/usrman/index.html). 

This code requires each node to be on an ad-hoc network and able to keyless ssh into the others; you can follow [these](https://geekytheory.com/tutorial-raspberry-pi-wireless-ad-hoc-network/) to set up an ad-hoc network and [these](http://superuser.com/questions/8077/how-do-i-set-up-ssh-so-i-dont-have-to-type-my-password) to set up keyless ssh'ing. Remember to add each node's hardware address (located in *cd /sys/class/net/wlan0/address*) to the others' page tables.

### CloudKSVD.py

**Usage:** `mpiexec -n <N> --hostifle <./hostfile> python CloudkSVD.py <tD> <t0> <tc> <tp>`

- Used for performance testing Cloud K-SVD. 
- Set custom node indexes and edges [to create a graph](https://mpi4py.scipy.org/docs/apiref/mpi4py.MPI.Graphcomm-class.html) that resembles your network before running
- Calls on MNIST_Loader.py and MainMethods.py

### MainMethods.py

- Contains functions for OMP, Cloud K-SVD, Active Dictionary Learning, degree discovery, and time synchronization
- Each function can be called individually as well after importing (`from MainMethods import *`) if ConsensusMethods.py is available

### ConsensusMethods.py

- Contains data transmission methods, Corrective Consensus, and the [Distributed Power Method](http://arxiv.org/pdf/1105.1185v1.pdf)
- Data transmission and reception is done at each node by polling for data within a given time out period
- Consensus requries weights, which requires the degree matrix generated through DiscoverDegrees in MainMethods.py

### MNIST_Loader.py

- Uses python-mnist library to load MNIST dataset into numpy matricies
- Loading is done through random sampling without replacement of the desired digit classes and amounts for each
- Random sampling may result in highly variable load times that could cause nodes in network to lose sync; use the barrier command to get around this
- The largest resolution available is 28x28; images may be scaled down but not up through the resize function
- By default, MNIST images are imported from a folder called *`python-mnist/data`* in the same folder as the program

### Debug.py

**Usage:** `mpiexec -n 5 --hostifle <./hostfile> python Debug.py`

- Change the option variable in the code to debug data transmission, consensus, or the power method
- By default, set to a fully connected 5 node network with little data

## Performance Data & Analysis

The code to import and analyze the data obtained from our performance testing is available as MATLAB code in the *Performance Data* folder. THe programs add each data point measuring the run time for Cloud K-SVD on the MNIST dataset for 6 different variables: Resolution (data dimension), Samples per Class (5 classes), Signals, Iterations of Cloud K-SVD, Iterations of Consensus, and Iterations of the Power Method. Further details and results are discussed in the paper. Running `Main.m` will import all the data points into MATLAB and organize them into a 6-D Matrix and a set of signals. Each signal contains the 6 independent variables followed by the run time. 
