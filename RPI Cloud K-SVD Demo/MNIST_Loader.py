from mnist import MNIST
import numpy as np
import pickle, os
import Image

# Functions used to load MNIST dataset onto the nodes

def resize(images,resolution):
    # Resizes a given set of images
    # ========================================================================
    # INPUT ARGUMENTS:
    #   images               (np.array) original images 
    #   resolution           (np.array) new resolution to resize images into
    # ========================================================================
    # OUTPUT: 
    #   resized              (np.array) resized images 
    # ========================================================================

    # Set up vars
    atoms = np.shape(images)[0]
    original_res = (int(np.sqrt(np.shape(images)[1])),int(np.sqrt(np.shape(images)[1])))
    image_length = resolution[0]*resolution[1]
    resized = np.array(np.zeros((image_length,atoms)))

    # Resize images
    for k in xrange(0,atoms):
        temp = (images[k].astype('uint8')).reshape(original_res)
        imtemp = Image.fromarray(temp)
        imtemp.thumbnail(resolution,Image.ANTIALIAS)
        temp = np.array(imtemp).astype('int').reshape((image_length))
        resized[0:(image_length),k] = temp.reshape((image_length))

    return resized


def organize(samples,labels,classes,amount):
    # Collects a given amount and set of classes from the MNIST dataset
    # ========================================================================
    # INPUT ARGUMENTS:
    #   samples              (np.array) images to sample from 
    #   labels               (np.array) labels corresponding to sample images
    #   classes              (np.array) classifications of images to collect
    #   amount               (int)      number of samples to collect for each  
    #                                   classification
    # ========================================================================
    # OUTPUT: 
    #   organized_samples    (np.matrix) samples from desired classes
    #   organized_labels     (np.array)  labels corresponding to the samples 
    # ========================================================================

    # Set up vars
    count = np.zeros(len(classes))   #keeps track of amount of samples per class
    atoms = np.shape(samples)[1]                    #number of samples
    dimension = np.shape(samples)[0]                #dimension
    filtered_samples = np.array(np.zeros((dimension,len(classes)*amount)))      
    filtered_labels  = np.array(np.zeros((len(classes)*amount)))       #empty arrays to fill

    # Randomly collect samples
    for k in np.random.permutation(atoms): 

        label_selected = labels[k] == np.array(classes)

        if np.all(count==amount): #if we have the required samples for each class
            break

        elif np.any(label_selected) and count[label_selected] < amount: # If we require another sample
            filtered_samples[:,int(sum(count))] = samples[:,k].reshape(dimension)  # Add the sample
            filtered_labels[int(sum(count))]  = labels[k].reshape(1)               # Add the label
            count += np.array(label_selected,dtype=int) #a Add it to the count of samples collected

    # Sort the samples by their label
    organized_samples = filtered_samples[:,filtered_labels.argsort()]
    organized_labels  = np.sort(filtered_labels)

    return np.matrix(organized_samples),organized_labels


def importMNIST(folder,resolution,classes,amount,signals):
    # Imports the original MNIST dataset from either a .pkl or their original
    # file format
    # ========================================================================
    # INPUT ARGUMENTS:
    #   folder               (string)   location of the data
    #   resolution           (np.array) resolution of the images
    #   classes              (np.array) image classifications to import
    #   amount               (int)      number of images to import per class
    #                                   for the dictionary
    #   signals              (int)      number of random images to import for
    #                                   the signal matrix Y
    # ========================================================================
    # OUTPUT: 
    #   D                    (np.matrix) dictionary of training images
    #   D_labels             (np.array)  labels corresponding to each atom in
    #                                    the dictionary
    #  Y                     (np.matrix) testing images to reproduce
    #  Y_labels              (np.array)  labels corresponding to each signal 
    # ========================================================================

    # If we are importing from a pickled file
    print 'importing MNIST data...'
    if os.path.isfile('saved_DY.pkl'):
        print 'found file'
        f = open('saved_DY.pkl','r')
        D = pickle.load(f)
        D_labels = pickle.load(f)
        Y = pickle.load(f)
        Y_labels = pickle.load(f)

        return np.matrix(D),D_labels,np.matrix(Y),Y_labels

    # Otherwise, importing from original dataset
    mndata = MNIST(folder)
    train_ims,train_labels = mndata.load_training()
    print 'training loaded'
    test_ims,test_labels = mndata.load_testing()
    print 'testing loaded'

    # Import samples for dictionary
    training_samples = resize(np.array(train_ims),resolution)
    training_labels = np.array(train_labels)
    D,D_labels = organize(training_samples,training_labels,classes,amount)
    print 'dictionary, D, made'

    random_idx = np.array(np.random.permutation(10000))[0:signals] #10000 is total signals avail

    # Collect random samples for the signal matrix
    Y = (resize(np.array(test_ims),resolution))[:,random_idx]
    Y_labels = np.array(test_labels)[random_idx]
    print 'signals, Y, made'

    # Save collected samples to a pickled file
    saveToFile(D,D_labels,Y,Y_labels)

    return np.matrix(D),D_labels,np.matrix(Y),Y_labels


def showImage(images,k,resolution):
    # Displays the k^th image at a given resolution from the matrix 'images'

    temp = (images[0:(resolution[1]*resolution[0]),k].astype('uint8'))
    imtemp = Image.fromarray(temp.reshape((resolution[0],resolution[1])))
    imtemp.show()


def saveToFile(D,D_labels,Y,Y_labels):
    # Pickles a dictionary, signal matrix, and their labels

    f = open('saved_DY.pkl', 'w')
    pickle.dump(D,f)
    pickle.dump(D_labels,f)
    pickle.dump(Y,f)
    pickle.dump(Y_labels,f)
    f.close()

