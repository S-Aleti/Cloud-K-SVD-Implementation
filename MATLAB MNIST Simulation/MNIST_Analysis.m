
% ========================================================================
%
% The purpose of this code is to simulate KSVD in a local and distributed 
% network using MNISt data and determine the average classification error 
% of each algorithm through a given amount of Monte-Carlo trials
% 
% This code can be run with the following paramters in approximately 5 min
% A text output is given in the command window and the data for each MC
% trial's accuracy is stored in KSVDacc and cKSVDacc
%
% ========================================================================

%% Parameters

trials = 3;  % MC trials

% Select smaller number of samples from full data set
scaling = (144/784);    % scaling factor
nums = [0,3,5,8,9];     % MNIST numbers to use
trainsamples = 100;     % training samples 
testsamples = 100;      % testing samples

%KSVD params
T0 = 10;
Td = 10;

%Cloud params
N = 10; % number of nodes, must be a factor of train & test samples
Tc = 5;
Tp = 5;

%% Main Data Prep

% Load up data
[images,labels] = MNISTload('train-images.idx3-ubyte',...
                                'train-labels.idx1-ubyte');
disp('Training images and labels loaded');
[testimages,testlabels] = MNISTload('t10k-images.idx3-ubyte',...
                                        't10k-labels.idx1-ubyte');
disp('Testing images and labels loaded');
disp('------------------------------');

% Determine max samples that can be taken from data
maxtrain = zeros(1,length(nums));
for l = 1:length(labels)
    maxtrain = maxtrain + (labels(l)==nums);
end

% Do same for test images
maxtest = zeros(1,length(nums));
for l = 1:length(testlabels)
    maxtest = maxtest + (testlabels(l)==nums);
end

% Create primary sample dictionary
disp('Creating main sample dictionary...'); 
[images,labels,~] = CollectSamples(images,labels,scaling,...
                                    nums,min(maxtrain));
[testimages,testlabels,~] = CollectSamples(testimages,testlabels,...
                                            scaling,nums,min(maxtest));

%% Monte Carlo Trials
for MCT = 1:trials
    
    disp('==============================');
    disp(['-----------Trial ' num2str(MCT) '-----------']);
    disp('==============================');
    
    %% KSVD Data Prep
    % Prepare files from selected numbers
    disp('Preparing training images...');
    [trainIm,trainLa,trainInd] = CollectSamples(images,labels,...
                                                  1,nums,trainsamples);
    disp('Preparing test images...');
    [testIm,testLa,testInd] = CollectSamples(testimages,testlabels,1,...
                                                nums,testsamples);
    
    %% KSVD
    disp('==============================');
    disp('   Testing Centralized KSVD');
    disp('==============================');
    Y = testIm;
    D = trainIm;
    [localD,localx,error] = CentralKSVD(Y,D,T0,Td);
    
    %% Analyze KSVD Residuals
    r = [];
    for s = 1:size(testIm,2)
        for c = 1:5 % since 5 numbers analyzed
            range = [trainInd(c):trainInd(c+1)-1];
            r(s,c) = norm(Y(:,s)-localD(:,range)*localx(range,s));
        end
    end
    
    [~,ind] = min(r');
    guesses = nums(ind)';
    actual  = testLa;
    correct = (nums(ind)')==testLa;
    overall_accuracy = mean(correct);
    Am = zeros(1,length(nums));
    
    for k = 1:length(correct)
        for n = 1:length(nums)
            if actual(k) == nums(n)
                Am(n) = Am(n) + correct(k);
            end
        end
    end
    
    number_accuracy = Am./testsamples;
    disp(['Overall accuracy is: ' num2str(overall_accuracy)]);
    Number_accuracy_is = [nums', number_accuracy']
    KSVDacc(:,:,MCT) = number_accuracy';
    
    disp('------------------------------');
    
    %% Cloud Data Prep
    atoms = trainsamples/N;
    newInd = ((trainInd-1)./2)+1;
    disp(['Preparing cloud data...']);
    
    % Set up cloud dictionary
    cloudD= [];
    for s = 1:N
        [cloudD(:,:,s),~,trainInd] = CollectSamples(images,labels,1,...
                                                     nums,trainsamples/N);
    end
    
    % Set up cloud signals
    cloudY= [];
    solutions=[];
    for s = 1:N
        [cloudY(:,:,s),solutions(:,:,s),testInd] = CollectSamples(...
                                testimages,testlabels,1,nums,testsamples);
    end
    
    %% Cloud KSVD
    
    disp('==============================');
    disp('      Testing Cloud KSVD');
    disp('==============================');
    
    T0;% sparsity constraint
    Td;% number of iterations
    [newcloudD,nodeX,error] = CloudKSVD(cloudY,cloudD,T0,Td,Tc,Tp);
    
    %% Analyze cKSVD Residuals
    
    r =[];
    for node = 1:N
        for s = 1:size(cloudY,2)
            for c = 1:length(nums)
                range = [trainInd(c):trainInd(c+1)-1];
                r(s,c,node) = norm(cloudY(:,s,node)-...
                            newcloudD(:,range,node)*nodeX(range,s,node));
            end
        end
    end
    
    residueClass = r;
    
    for nodes = 1:N
        
        r = residueClass(:,:,nodes);
        [~,ind] = min(r');
        guesses = nums(ind)';
        actual = solutions(:,:,nodes);
        correct = (nums(ind)')==(solutions(:,:,nodes));
        node_accuracy(:,:,nodes) = mean(correct);
        Am = zeros(1,length(nums));
        
        for k = 1:length(correct)
            for j = 1:length(nums)
                if actual(k) == nums(j)
                    Am(j) = Am(j) + correct(k);
                end
            end
        end
        
        number_accuracy(nodes,:) = Am./(testsamples);
        
    end
    
    Overall_accuracy_is = mean(mean(number_accuracy))
    NumberAccuracy_is = [nums',mean(number_accuracy,1)']
    disp('------------------------------');
    cKSVDacc(:,:,MCT) = mean(number_accuracy,1)';
    
end

%% Results

disp(' ');
disp('==============================');

Overall_KSVD_Accuracy = [nums',mean(KSVDacc,3)]
Mean_KSVD_Accuracy = mean(mean(KSVDacc,3))

disp('------------------------------');

Overall_cKSVD_Accuracy = [nums',mean(cKSVDacc,3)]
Mean_cKSVD_Accuracy = mean(mean(cKSVDacc,3))

disp('==============================');
    
%% Plot Data

plot(mean(KSVDacc,3),'bs-');
hold on;
plot(mean(cKSVDacc,3),'ro-');

set(gca,'XTickLabel',cellfun(@num2str,num2cell(nums(:)),'uniformoutput'...
    ,false),'XTick',1:5);

ylabel('Detection Rate');
xlabel('Digits');
legend('Central K-SVD','Cloud K-SVD')


