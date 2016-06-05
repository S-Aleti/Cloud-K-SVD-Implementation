%% MNIST Image Loading Functions

function [images,labels] = MNISTload(images_location,labels_location)
% MNISTLOAD loads images from MNIST files (.idx#-ubyte) into a dictionary
% ========================================================================
% INPUT ARGUMENTS:
%   images_location       (string) location, file, of the MNIST image data
%   labels_location       (string) location, file, of the MNIST label data
% ========================================================================
% OUTPUT: 
%   images                (matrix) processed images in matrix form
%   labels                (matrix) processed labels in matrix form
% ========================================================================   

%% Main 

%load images
images = loadMNISTImages(images_location);

%load labels
labels = loadMNISTLabels(labels_location);

end

function images = loadMNISTImages(filename)
% LOADMNISTIMAGES converts the files of MNIST images into matrix form
% ========================================================================  

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Bad magic number in ', filename, '']);

numImages = fread(fp, 1, 'int32', 0, 'ieee-be');
numRows = fread(fp, 1, 'int32', 0, 'ieee-be');
numCols = fread(fp, 1, 'int32', 0, 'ieee-be');

images = fread(fp, inf, 'unsigned char');
images = reshape(images, numCols, numRows, numImages);
images = permute(images,[2 1 3]);

fclose(fp);

% Reshape to #pixels x #examples
images = reshape(images, size(images, 1) * size(images, 2), size(images, 3));
% Convert to double and rescale to [0,1]
images = double(images) / 255;

end

function labels = loadMNISTLabels(filename)
% LOADMNISTLABELS converts the files of MNIST labels into matrix form
% ========================================================================  

fp = fopen(filename, 'rb');
assert(fp ~= -1, ['Could not open ', filename, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Bad magic number in ', filename, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');

labels = fread(fp, inf, 'unsigned char');

assert(size(labels,1) == numLabels, 'Mismatch in label count');

fclose(fp);

end
