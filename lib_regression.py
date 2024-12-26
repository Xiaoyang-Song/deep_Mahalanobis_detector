# several functions are from https://github.com/xingjunm/lid_adversarial_subspace_detection
from __future__ import print_function
import numpy as np
import os
import calculate_log as callog
from collections import Counter

from scipy.spatial.distance import pdist, cdist, squareform


def block_split(X, Y, out, n_test=100000):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    if out == 'svhn':
        partition = 26032
    elif out == 'DTD':
        partition = 5640
    elif out == 'LSUN-C':
        partition = 10000
    elif out == 'LSUN-R':
        partition = 10000
    elif out == 'Places365-small':
        partition = 36500
    elif out == 'iSUN':
        partition = 8925
    elif out == 'fm89':
        partition = 2000
    elif out == 'svhn89':
        partition = 3255
    elif out == 'mnist17':
        partition = 1983
    elif out == 'fm':
        partition = 5000
    elif out == 'cifar10':
        partition = 5000
    elif out == 'imagenet-c':
        partition = 5000

    partition = min(partition, n_test)

    # Structure: OOD + IND
    # print(Counter(Y))
    X_adv, Y_adv = X[:partition], Y[:partition]
    print(Counter(Y_adv))
    X_norm, Y_norm = X[partition: :], Y[partition: :]
    print(Counter(Y_norm))
    num_train = 1000

    X_train = np.concatenate((X_norm[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_adv[:num_train]))
    print(Counter(Y_train))

    X_test = np.concatenate((X_norm[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_adv[num_train:]))
    print(Counter(Y_test))

    return X_train, Y_train, X_test, Y_test


def block_split_adv(X, Y):
    """
    Split the data training and testing
    :return: X (data) and Y (label) for training / testing
    """
    num_samples = X.shape[0]
    partition = int(num_samples / 3)
    X_adv, Y_adv = X[:partition], Y[:partition]
    X_norm, Y_norm = X[partition: 2*partition], Y[partition: 2*partition]
    X_noisy, Y_noisy = X[2*partition:], Y[2*partition:]
    num_train = int(partition*0.1)
    X_train = np.concatenate((X_norm[:num_train], X_noisy[:num_train], X_adv[:num_train]))
    Y_train = np.concatenate((Y_norm[:num_train], Y_noisy[:num_train], Y_adv[:num_train]))

    X_test = np.concatenate((X_norm[num_train:], X_noisy[num_train:], X_adv[num_train:]))
    Y_test = np.concatenate((Y_norm[num_train:], Y_noisy[num_train:], Y_adv[num_train:]))

    return X_train, Y_train, X_test, Y_test

def detection_performance(regressor, X, Y, outf):

    """
    Measure the detection performance
    return: detection metrics
    """
    num_samples = X.shape[0]
    l1 = open('%s/confidence_TMP_In.txt'%outf, 'w')
    l2 = open('%s/confidence_TMP_Out.txt'%outf, 'w')
    y_pred = regressor.predict_proba(X)[:, 1]

    for i in range(num_samples):
        if Y[i] == 0:
            l1.write("{}\n".format(-y_pred[i]))
        else:
            l2.write("{}\n".format(-y_pred[i]))
    l1.close()
    l2.close()
    results = callog.metric(outf, ['TMP'])
    return results



    
def load_characteristics(score, dataset, out, outf):
    """
    Load the calculated scores
    return: data and label of input score
    """
    X, Y = None, None
    
    file_name = os.path.join(outf, "%s_%s_%s.npy" % (score, dataset, out))
    data = np.load(file_name)
    
    if X is None:
        X = data[:, :-1]
    else:
        X = np.concatenate((X, data[:, :-1]), axis=1)
    if Y is None:
        Y = data[:, -1] # labels only need to load once
         
    return X, Y