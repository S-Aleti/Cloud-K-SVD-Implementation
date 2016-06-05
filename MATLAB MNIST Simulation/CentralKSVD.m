function [D,x,error] = CentralKSVD(Y,D,T0,Td)
% CENTRALKSVD runs the K-SVD algorithm on a given dictionary and set of
% signals, see paper: ieeexplore.ieee.org/document/1710377/
% ========================================================================
% INPUT ARGUMENTS:
%   Y                    (matrix) set of signals to represent
%   D                    (matrix) intial dictionary 
%   T0                   (scalar) sparsity of representation
%   Td                   (scalar) iterations of the K-SVD algorithm
% ========================================================================
% OUTPUT: 
%   D                    (matrix) new dictionary 
%   x                    (matrix) set of linear combinations to produce the
%                        given signals: D*x ~= Y
%   error                (matrix) representation error of Y (L1 norm)
% ========================================================================

%% Vars
D = normc(D);
K = size(D,2);

%% Algorithm
for iteration = 1:Td
    %% SparseCodingStage
    x = full(omp(D,Y,[],T0)); %input dictionary, signals, sparsity
    
    %% Codebook Update
    for k = 1:K
     
        wk = find(x(k,:));
        Ek = (Y-D*x)+D(:,k)*x(k,:); % quickly finds error matrix
        ERk = Ek(:,wk);
        
        if ~isempty(wk)
            % fast svd decomp, only grabs largest eigenvector
            [U1,S1,V1] = svds(ERk,1);
            D(:,k) = normc(U1);
            x(k,wk) = (S1*V1');
        end
        
    end
    
    error(iteration,:) = norm(Y-D*x);
    
end
end

