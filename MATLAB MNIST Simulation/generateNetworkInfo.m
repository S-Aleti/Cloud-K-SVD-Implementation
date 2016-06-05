function [ adj,W ] = generateNetworkInfo( size )
% GENERATENETWORKINFO creates a connected, undirected graph, its adjacency 
% matrix, and the Metropolis-Hastings weights that correspond to it
% ========================================================================  

adj = genNodes(size);    %Ardos Renier with varying connection chances
W   = genW(adj);         %Metropolis Hastings Weights

end

%% Generate Network Info Local Functions

function [ adjacencymatrix ] = genNodes( size )
% GENNODES creates a fully connected, sparse network of nodes
% ========================================================================  

P = ((2/size).^0.5);  %probability of connection chosen to consistently

if size == 1          %generate very sparsely connected graphs quickly
    
    adjacencymatrix = 1;
    return;
    
else
    
    A = zeros(size);
    
    for i = 1:size
        for j = 1:(i-1)      %nodes assumed to be connected to themselves
            if rand() < P    %biased "coin flip" to add nodes
                A(i,j) = 1;  %add bidirectional nodes since mesh network
                A(j,i) = 1;  %can send or receive data
            end
        end
    end
    
    if (~isConnected(A))
        A = genNodes(size);
    end
    
    adjacencymatrix = A;
    
end

end

function [boolean] = isConnected(adj)
% ISCONNECTED ensures a given network (adjacency matrix) is fully connected
% ========================================================================  

tempsum = zeros(size(adj,1));

for k = 1:length(adj)
    tempsum = tempsum + mpower(adj,k); %find walks
end                                    %check if there's a walk to every

boolean = 0==sum(sum(tempsum==0));     %node

end

function [ W ] = genW( adjacency )
% GENW generates the Metropolis-Hastings weights for a given network
% ========================================================================  

[rows,columns] = size(adjacency);
W = zeros(rows,columns);

%% Main

for i = 1:rows
    
    for j = 1:columns
        
        if (i~=j && adjacency(i,j)~=0)
            W(i,j) = 1/(1 + max(sum(adjacency(i,:)),sum(adjacency(:,j))));
        elseif (i == j)
            
            tempsum = 0;
            for x = 1:columns
                if (x~=i && adjacency(x,j)~=0)
                    tempsum = tempsum + 1/(1 + ...
                        max(sum(adjacency(x,:)),sum(adjacency(:,j))));
                end
            end
            
            W(i,j) = 1 - tempsum;
            
        else
            W(i,j) = 0;
        end
        
    end
    
end

end
