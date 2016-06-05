function [ newIm2,newLa,index ] = CollectSamples(images,labels,scaling, ...
                                                    numbers,samples )
% COLLECTSAMPLES converts a set of MNIST images and their classifications 
% into a usable set of signals for training and testing data
% ========================================================================
% INPUT ARGUMENTS:
%   images               (matrix) set of images in vector form
%   labels               (labels) classification index of each image
%   scaling              (matrix) percent to scale the image data by
%   numbers              (matrix) which image types (MNIST data set is
%                                 composed of images of numbers) to extract
%   samples              (scalar) number of images to extract for each type
% ========================================================================
% OUTPUT: 
%   newIm2                (matrix) processed images in matrix form
%   newLa                 (matrix) classification labels corresponding to 
%                                  each image
%   index                 (matrix) new indicies for images, only used for
%                                  debugging
% ========================================================================                                                
%% Prelims
newIm = zeros(size(images,1),samples*length(numbers));
newLa = zeros(samples*length(numbers),1);
SamplesMatrix = samples*ones(1,length(numbers));

%% Extraction 
order = randperm(length(labels));        
for iter = 1:length(labels)                    % parses image label file
    
    k = order(iter);                           % checks if desired class
    temp = find(labels(k)==numbers);           % adds image to new matrix
    
    if (sum(SamplesMatrix)==0)                 % stores image label to vec
        break;
    end
    
    if any(temp)
        
        if SamplesMatrix(temp) > 0
            newIm(:,SamplesMatrix(temp)+samples*(temp-1)) = images(:,k);
            newLa(SamplesMatrix(temp)+samples*(temp-1)) = labels(k);
            SamplesMatrix(temp) = SamplesMatrix(temp)-1;
        
        end
        
    end
    
end

% If there's not enough images for the demanded samples
assert((sum(SamplesMatrix)==0),['Not enough images available; missing '...
    num2str(sum(SamplesMatrix)) ' samples'])

%% Resizing Code
if scaling ~= 1                                  % changes scaling of image
                                                 % must be square output
    imRes = size(images,1).^0.5;
    newRes = round((imRes^2 * scaling)^0.5);
    assert(abs((imRes^2 * scaling)^0.5 - newRes)<1e-3,['Inappropriate scaling.'])
    
    for k = 1:length(newLa)
        resizedImage = imresize(reshape(newIm(:,k),imRes,imRes),sqrt(scaling));
        reshapedImage = reshape(resizedImage,newRes*newRes,1);
        newIm2(:,k) = reshapedImage';
    end
    
else
    newIm2 = newIm;
end

index = [0:samples:(length(numbers)*samples)] + 1;

end

