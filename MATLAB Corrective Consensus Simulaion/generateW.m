function [ W ] = generateW( adjacency )
% GENERATEW creates a matrix of Metropolis-Hastings weights based on a 
% given network of nodes, see paper: 
% ieeexplore.ieee.org/iel7/6844297/6853544/06854643.pdf
% ========================================================================
% INPUT ARGUMENTS:
%   adjacency                  (matrix) adjacency matrix of network
% ========================================================================
% OUTPUT: 
%   W                          (matrix) Metropolis-Hastings weights
% ========================================================================

[rows,columns] = size(adjacency);

W = zeros(rows,columns); 

%% Metropolis- Hastings Weights
% See paper for explanation

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
