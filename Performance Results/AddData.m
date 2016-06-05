function [DataMatrix,DataSignal] = AddData(DataMatrix,DataSignal,Resolution,...
    Classes,Amount_per_class,Signals,tD,t0,tc,tp,run_time)
% ADDDATA is used in main to transform the data collected into a matrix for
% further analysis
% ========================================================================
% INPUT ARGUMENTS:
%   DataMatrix           (matrix) six-dimensional, contains the following 
%                                 data: 'Pixels','Amount','Signals','tD',
%                                 'tc','tp','run_time'
%   DataSignal           (matrix) reformatted DataMatrix in two dimensions
%   Resolution           (matrix) contains height and width of images in
%                                 pixels
%   Classes              (matrix) list of classifications being analyzed
%   Amount_per_class     (matrix) training samples used per class
%   Signals              (scalar) total number of testing samples used
%   tD                   (scalar) iterations of Cloud K-SVD
%   t0                   (scalar) sparsity constraint used
%   tc                   (scalar) iterations of the consensus algorithm
%   tp                   (scalar) iterations of the power method
% ========================================================================

%% Formatting

Pixels = Resolution(1)*Resolution(2);
%Set each variable to an index to add to the 6D matrix
res_dim= find([36,100,256]==Pixels); %make new dimension later (2nd)
amount_dim= find([5,10]==Amount_per_class);  
sig_dim = find([10,25]==Signals);
tD_dim = find([3,5]==tD);
tc_dim = find([3,5]==tc);
tp_dim = find([3,5]==tp);   

%% Add to matrix

DataMatrix(res_dim,amount_dim,sig_dim,tD_dim,tc_dim,tp_dim) = run_time;

%% For feature analysis, adds each sample as a 7x1 vector "trial"

New_Signal = [Pixels,Amount_per_class,...
    Signals,tD,tc,tp,run_time]';
DataSignal = [DataSignal,New_Signal];

end

