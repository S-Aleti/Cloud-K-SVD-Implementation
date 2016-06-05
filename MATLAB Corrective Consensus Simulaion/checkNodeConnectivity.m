function [ connected ] = checkNodes( adj )
% CHECKNODES determines if a graph is fully connected
% ========================================================================
% INPUT ARGUMENTS:
%   adj                  (matrix) adjacency matrix of nodes
% ========================================================================
% OUTPUT: 
%   connected            (boolean) 
% ========================================================================

[rows,columns] = size(adj);
tempsum = adj;

for i = 2:rows
    % distances from each node to another
    tempsum = tempsum + mpower(adj,i);
end

% True when all nodes have paths to each other
connected = (sum(tempsum(:)==0) == 0);  

end

