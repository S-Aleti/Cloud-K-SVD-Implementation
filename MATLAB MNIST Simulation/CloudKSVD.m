
function [nodeD,nodeX,error] = CloudKSVD(cloudY,cloudD,T0,Td,Tc,Tp)
% CLOUDKSVD runs the Cloud K-SVD algorithm on a set of nodes that have
% preliminary dictionaries used to represent their local samples; the algo
% finds the best global dictionary to represent all local samples, see 
% paper: https://arxiv.org/abs/1412.7839
% ========================================================================
% INPUT ARGUMENTS:
%   cloudY               (matrix) three dimensional matrix containing each 
%                                 nodes' signals 
%   cloudD               (matrix) three dimensional matrix containing each
%                                 node's initial dictionary 
%   T0                   (scalar) sparsity of representation
%   Td                   (scalar) iterations of the K-SVD algorithm
%   Tc                   (scalar) iterations of consensus algo
%   Tp                   (scalar) iterations of distributed power method
% ========================================================================
% OUTPUT: 
%   nodeD                (matrix) new dictionary at each node 
%   nodeX                (matrix) set of linear combinations at each node
%                                 to reproduce their local signals
%   error                (matrix) representation error of local signals at
%                                 every node
% ========================================================================

%% Prelims
%variables are same names as described in the paper
D = cloudD;
Y = cloudY;
n = size(D,1);
K = size(D,2);
N = size(D,3);


%% Assign Nodes Local Data and Dictionaries
nodeD = D;
nodeY = Y;
%Graph info
[adj,W] = generateNetworkInfo(N);   %adjacency matrix
dref = rand(n,1);                   %common reference direction

%% Cloud K-SVD

for Iteration = 1:Td
    %% Sparse Coding Stage
    for i = 1:N
        nodeD(:,:,i) = normc(nodeD(:,:,i));
        nodeX(:,:,i) = full(omp(nodeD(:,:,i),nodeY(:,:,i),[],T0));
    end
    
    %% Codebook Update Stage
    for k = 1:K
        
        for i = 1:N
            wk = find(nodeX(k,:,i)); %fast calc of Error below
            Ek = (nodeY(:,:,i)-nodeD(:,:,i)*nodeX(:,:,i))+nodeD(:,k,i)*nodeX(k,:,i);
            EkR= Ek(:,wk);           %selecting relevent error columns
            M(:,:,i) = EkR*EkR';     %find M
        end
        
        %% Cloud update
        q = PowerConsensus(M,adj,Tc,Tp);
        M = [];                      %empty M, and define q(:,eachNode)
        
        for i = 1:N
            
            wk = find(nodeX(k,:,i));
            Ek = (nodeY(:,:,i)-nodeD(:,:,i)*nodeX(:,:,i))+...
                nodeD(:,k,i)*nodeX(k,:,i);
            EkR= Ek(:,wk);           %Recalculate since we can't store
            
            if length(wk) ~= 0
                
                nodeD(:,k,i) = sign(dot(q(:,i),dref))*(q(:,i));
                nodeX(k,:,i) = 0;                    %clean x
                nodeX(k,wk,i) = nodeD(:,k,i)'*EkR;   %update x
                
            end
            
            error(Iteration,i) = norm(nodeY(:,:,i)-nodeD(:,:,i)*nodeX(:,:,i));
            
        end
        
    end
    
end

end

%% Distributed Power Method 
function [q] = PowerConsensus(M,W,tc,tp)
% POWERCONSENSUS runs distributed power method on a given set of nodes and
% data and returns the results at each node
% ========================================================================
% INPUT ARGUMENTS:
%   M                    (matrix) three dimensional matrix containing each 
%                         nodes' data
%   W                    (matrix) Metropolis-Hastings weights for each node
%   tc                   (scalar) iterations of consensus algo
%   tp                   (scalar) iterations of distributed power method
% ========================================================================
% OUTPUT: 
%   q                    (matrix) new data at each node 
% ========================================================================

%% Intialize data
nodes = size(M,3);
datasize = size(M,1);
q = ones(datasize,nodes);

%% Simulate Consensus Method
newW = mpower(W,tc);

%% Simulate Power method 

if isempty(find(sum(M,3)))         %Check if there is any error
    
    q = zeros(datasize,nodes);     %If there is not, all vecs are eig vecs
    
else
    
    for powermethod = 1:tp
        
        for i = 1:nodes
            z(:,i) = M(:,:,i)*q(:,i);
        end
        
        v = z*newW;
        q = normc(v);             %q is normalized locally
        
    end
    
end

end