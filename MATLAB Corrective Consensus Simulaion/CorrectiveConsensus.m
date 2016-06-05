function [x,runningAvg] = CorrectiveConsensus(n,dim,packetloss,iterations,k)
% CORRECTIVECONSENSUS simulates the corrective consensus algorithm which
% is provided here: ieeexplore.ieee.org/iel5/5707200/5716927/05717925.pdf
% in a distributed network with stochastic packet loss
% ========================================================================
% INPUT ARGUMENTS:
%   n                    (scalar) number of nodes
%   dim                  (scalar) size of each node's random data vector
%   packetloss           (scalar) mean packets dropped per iteration
%   k                    (scalar) number of standard iterations before
%                                 a corrective iteration
%   iterations           (scalar) total iterations of the consensus algo
%   
% ========================================================================
% OUTPUT: 
%   x                    (matrix: data, node, iteration) value of each
%                                 nodes data at each iteration
%   runningAvg           (matrix) average all data at each iteration, the
%                                 goal is to reach a 'consensus' on the 
%                                 average of every node's data
% ========================================================================

%% Generate nodes and data

Adj = generateNodes(n);             % Random distributed network
Adj = Adj+diag(ones(1,n));
Edges = find(sparse(Adj));          
W   = generateW(Adj);               % Metropolis-Hastings weights
x = rand(dim,n)*10;                 % Node Data
phi = zeros(n,n,dim);               % Mass change data (see paper)
v = phi;                            % Phi tracker (see paper)
runningAvg = mean(x(:,:,1),2);
l = 1;                              % Starting teration index

%% Corrective Consensus

for t = 1:iterations
    %% Stochastic Packet Loss
    Adj_proper = Adj;
    packets_lost = (poissrnd(packetloss));
    disp([num2str(packets_lost) ' Packets lost at t: ' num2str(t)])
    
    for packetloss = 1:packets_lost
        location = Edges(ceil(rand()*length(Edges)));
        Adj(location) = 0;
    end
    
    if t ~= l*(k+1)-1
    %% Standard Iteration
        for i = 1:n
            summation = 0;
            for j = 1:n
                phi(i,j,:,t+1) = phi(i,j,:,t); % carry over previous mass
                if Adj(i,j) == 1
                    difference= x(:,j,t)-x(:,i,t);
                    summation = summation + W(i,j)*difference;       % mass from other nodes
                    phi(i,j,:,t+1) = phi(i,j,:,t+1) + reshape(W(i,j)*difference,1,1,dim,1); %add new mass
                end
            end
            x(:,i,t+1)   = x(:,i,t)+summation; % update local data
        end
        
        
    else
    %% Corrective Iteration
        Delta = zeros(n,n,dim);
        for i = 1:n
            for j = 1:n
                if Adj(j,i)==1
                    Delta(i,j,:) = phi(i,j,:,t) + phi(j,i,:,t);   % find difference in mass distribution
                    v(i,j,l) = 1;
                end
                phi(i,j,:,t+1) = phi(i,j,:,t)-(0.5)*Delta(i,j)*v(i,j,l); % remove difference locally (2)
            end
            x(:,i,t+1) = x(:,i,t) - reshape((0.5)*sum(Delta(i,:,:).*...
                repmat(v(i,:,l),1,1,dim)),dim,1,1);  % compensate for difference (1)
        end  % logically, we compensate (1) and then remove (2), but code is faster other way around
        l = l+1; % update number of corrective iterations
    
    end
    
    Adj = Adj_proper; % reset to original adjacency matrix/connections
    runningAvg(:,t+1) = mean(x(:,:,t+1),2); % average of current data, goal is to keep it the same
                                            % as the initial/true average before packet loss
end

%% Plot results

plot([0:t],reshape(x(1,:,1:t+1),n,t+1)')
line([0,iterations],[runningAvg(1,1),runningAvg(1,1)],...
        'LineWidth',2,'LineStyle','--')

for marks = 1:floor(iterations/(k+1))
    line([marks*(k+1)-1,marks*(k+1)-1],1.1*[min(min(min(x))),...
           max(max(max(x)))],'LineStyle',':','Color',[0.1,0.1,0.1])
end

hold on;
plot([0:t],[runningAvg(1,:)],'r--','LineWidth',2)
axis([0,t,min(min(min(x))),max(max(max(x)))])
hold off;

end