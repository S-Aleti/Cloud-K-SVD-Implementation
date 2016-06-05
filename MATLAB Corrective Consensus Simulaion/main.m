%% Simulate Corrective Consensus

n = 15;              % nodes
dim = 20;            % dimension of each node's data vector
packetloss = 2;      % average number of nodes dropping data per iteration
iterations = 40;     % total number of consensus iterations
k = 20;              % number of iterations between each correction

[x, runningAvg] = CorrectiveConsensus(n,dim,packetloss,iterations,k);
