function [ adjacencymatrix ] = generateNodes( size )
% GENERATENODES creates a random, fully connected distributed network of
% notes represented by an adjacency matrix
% ========================================================================
% INPUT ARGUMENTS:
%   size                       (scalar) number of nodes in the network
% ========================================================================
% OUTPUT: 
%   adjacencymatrix            (matrix) adjacency matrix of nodes
% ========================================================================

%% Generate Nodes

P = ((2/size).^0.6);  %probability of connection chosen to consistently

% Network of size 1 is just one node connected to itself
if size == 1          
    
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
    
    % Ensure graph is fully connected or generate a new matrix
    if (~checkNodeConnectivity(A))
        A = generateNodes(size);
    end
    
    adjacencymatrix = A;
    
end

end
