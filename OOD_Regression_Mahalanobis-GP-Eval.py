"""
Created on Sun Oct 21 2018
@author: Kimin Lee
"""
from __future__ import print_function
import numpy as np
import os
import lib_regression
import argparse

from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


parser = argparse.ArgumentParser(description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--ind_dset', required=True, help='resnet | densenet')
parser.add_argument('--nf', type=int, default=None, help='n_features')
parser.add_argument('--n_test', type=int, default=None, help='n_test')
args = parser.parse_args()
print(args)


def main():
    # initial setup
    ind_dset = args.ind_dset
    dataset_list = [ind_dset]
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005',
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']
    # score_list = ['Mahalanobis_0.0']

    # train and measure the performance of Mahalanobis detector
    # Evaluation
    TPR=0.95
    n_test = args.n_test
    maha_ind_acc = [[], [], [], [], [], []]
    maha_ood_acc = [[], [], [], [], [], []]
    maha_auroc = [[], [], [], [], [], []]
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = './output/' + args.net_type + '_' + dataset + '_' + str(args.nf) + '/'

        if dataset == 'mnist':
            out_list = ['fm', 'svhn', 'imagenet-c', 'cifar10']
        elif dataset == 'imagenet10':
            out_list = ['DTD', 'LSUN-C', 'LSUN-R', 'Places365-small', 'iSUN', 'svhn']
            # out_list = ['DTD']

        for idx, out in tqdm(enumerate(out_list)):
            print('Out-of-distribution: ', out)
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(total_X, total_Y, out, n_test)
                # Train logistic regression classifier on validation set
                lr = LogisticRegressionCV(n_jobs=-1, max_iter=100000).fit(X_val, Y_val)
                # Find threshold on validation set
                num_samples = X_val.shape[0]
                ind_val = []
                y_pred = lr.predict_proba(X_val)[:, 1]
                for i in range(num_samples):
                    if Y_val[i] == 0:
                        ind_val.append(-y_pred[i])
                print(len(ind_val))
                threshold = np.quantile(ind_val, 1 - TPR)
                # Generate logistic regression scores
                lib_regression.detection_performance(lr, X_test, Y_test, outf)
                ind_test = np.loadtxt('%s/confidence_TMP_In.txt'%outf, delimiter=' ')
                ood_test = np.loadtxt('%s/confidence_TMP_Out.txt'%outf, delimiter=' ')

                print(f"Testing set size: {len(ind_test)}, {len(ood_test)}")
                ind_acc = sum(ind_test >= threshold) / len(ind_test)
                ood_acc = sum(ood_test < threshold) / len(ood_test)
                # AUROC calculation
                scores = np.concatenate((ind_test, ood_test))  # Combine the arrays
                labels = np.concatenate((np.ones(ind_test.shape[0]), np.zeros(ood_test.shape[0])))  # Labels: 0 for InD, 1 for OoD
                # Calculate AUROC
                auroc = roc_auc_score(labels, scores)

                # Append results
                maha_ind_acc[idx].append(ind_acc)
                maha_ood_acc[idx].append(ood_acc)
                maha_auroc[idx].append(auroc)


    for count_out, out in enumerate(out_list):
        print(f"OOD dataset: {out}")
        print('IND ACC')
        print(100*np.round(maha_ind_acc[count_out],2))
        print(100*np.round(np.mean(maha_ind_acc[count_out]), 2))
        print("OOD ACC")
        print(100*np.round(maha_ood_acc[count_out],2))
        print(100*np.round(np.mean(maha_ood_acc[count_out]), 2))
        print("AUROC")
        print(100*np.round(maha_auroc[count_out],2))
        print(100*np.round(np.mean(maha_auroc[count_out]), 2))




if __name__ == '__main__':
    main()
