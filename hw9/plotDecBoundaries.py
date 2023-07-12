################################################
## EE559 HW1, Prof. Jenkins
## Created by Arindam Jati
## Tested in Python 3.6.3, OSX El Capitan, and subsequent versions
################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def plotDecBoundaries(means, func=None, fsize=(6,6), title=None, plt_mean=True, plt_dec=True, plt_data=False, training=None, label_train=None):

    '''
    Plot the decision boundaries and data points for minimum distance to
    class mean classifier

    training: traning data, N x 2 matrix:
        N: number of data points
        d: number of features
        if d > 2 then the first and second features will be plotted (1st and 2nd column (0 and 1 index))
    label_train: class lables correspond to training data, N x 1 array:
        N: number of data points
        the labels should start numbering from 1 (not 0)
        code works for up to 3 classes
    sample_mean: mean vector for each class, C x d matrix:
        C: number of classes
        each row of the sample_mean matrix is the coordinate of each sample mean
    '''

    #
    # Total number of classes
    nclass = 2

    # Set the feature range for ploting
    max_x = 4
    min_x = -8
    max_y = 6
    min_y = -6

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.05

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape

    xy = np.hstack( (x.reshape(x.shape[0]*x.shape[1], 1, order='F'), y.reshape(y.shape[0]*y.shape[1], 1, order='F')) ) # make (x,y) pairs as a bunch of row vectors.

    plt.figure(figsize=fsize)

    pred_label = func(xy)

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    if plt_dec:
        # show the image, give each coordinate a color according to its class label
        plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower', aspect='auto')


    if plt_data:
        # plot the class training data.
        plt.plot(training[label_train == 1, 0], training[label_train == 1, 1], 'r.')
        plt.plot(training[label_train == 2, 0], training[label_train == 2, 1], 'g.')
        l = plt.legend(('Class 1', 'Class 2'), loc=2)
        plt.gca().add_artist(l)


    if plt_mean:
        plt.scatter(means[0, 0], means[0, 1])
        plt.scatter(means[1, 0], means[1, 1])
        plt.scatter(means[2, 0], means[2, 1])

        # include legend for training data
        l = plt.legend(('Mean 1', 'Mean 2', 'Mean3'), loc=2)
        plt.gca().add_artist(l)


    plt.title(title)


    plt.show()
