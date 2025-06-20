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

parser = argparse.ArgumentParser(
    description='PyTorch code: Mahalanobis detector')
parser.add_argument('--net_type', required=True, help='resnet | densenet')
parser.add_argument('--ind_dset', required=True, help='resnet | densenet')
args = parser.parse_args()
print(args)


def main():
    # initial setup
    ind_dset = args.ind_dset
    dataset_list = [ind_dset]
    score_list = ['Mahalanobis_0.0', 'Mahalanobis_0.01', 'Mahalanobis_0.005',
                  'Mahalanobis_0.002', 'Mahalanobis_0.0014', 'Mahalanobis_0.001', 'Mahalanobis_0.0005']

    # train and measure the performance of Mahalanobis detector
    list_best_results, list_best_results_index = [], []
    for dataset in dataset_list:
        print('In-distribution: ', dataset)
        outf = './output/' + args.net_type + '_' + dataset + '/'
        out_list = ['svhn']
        if dataset == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        # MNIST Within-Dataset Experiment
        elif dataset == 'mnist07':
            out_list = ['mnist89']
        # FashionMNIST Within-Dataset Experiment
        elif dataset == 'fm07':
            out_list = ['fm89']
        # MNIST-FashionMNIST Between-Dataset Experiment
        elif dataset == 'mnist':
            out_list = ['fm']
        # SVHN Within-Dataset Experiment
        elif dataset == 'svhn07':
            out_list = ['svhn89']
        # CIFAR10-SVHN Between-Dataset Experiment
        elif dataset == 'cifar10':
            out_list = ['svhn']

        list_best_results_out, list_best_results_index_out = [], []
        for out in out_list:
            print('Out-of-distribution: ', out)
            best_tnr_95, best_result_95, best_index_95 = 0, 0, 0
            best_tnr_99, best_result_99, best_index_99 = 0, 0, 0
            for score in score_list:
                total_X, total_Y = lib_regression.load_characteristics(
                    score, dataset, out, outf)
                X_val, Y_val, X_test, Y_test = lib_regression.block_split(
                    total_X, total_Y, out)
                X_train = np.concatenate((X_val[:500], X_val[1000:1500]))
                Y_train = np.concatenate((Y_val[:500], Y_val[1000:1500]))
                X_val_for_test = np.concatenate(
                    (X_val[500:1000], X_val[1500:]))
                Y_val_for_test = np.concatenate(
                    (Y_val[500:1000], Y_val[1500:]))
                lr = LogisticRegressionCV(n_jobs=-1, max_iter=100000).fit(X_train, Y_train)
                y_pred = lr.predict_proba(X_train)[:, 1]
                #print('training mse: {:.4f}'.format(np.mean(y_pred - Y_train)))
                y_pred = lr.predict_proba(X_val_for_test)[:, 1]
                #print('test mse: {:.4f}'.format(np.mean(y_pred - Y_val_for_test)))
                results = lib_regression.detection_performance(
                    lr, X_val_for_test, Y_val_for_test, outf)
                
                if best_tnr_95 < results['TMP']['TNR95']:
                    best_tnr_95 = results['TMP']['TNR95']
                    best_index_95 = score
                    best_result_95 = lib_regression.detection_performance(
                        lr, X_test, Y_test, outf)
                    

            list_best_results_out.append(best_result_95)
            list_best_results_index_out.append(best_index_95)
        list_best_results.append(list_best_results_out)
        list_best_results_index.append(list_best_results_index_out)

    # print the results
    count_in = 0
    # mtypes = ['TNR', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    mtypes = ['TNR95', 'TNR99', 'AUROC', 'DTACC', 'AUIN', 'AUOUT']
    for in_list in list_best_results:
        print('in_distribution: ' + dataset_list[count_in] + '==========')
        out_list = ['svhn', 'imagenet_resize', 'lsun_resize']
        if dataset_list[count_in] == 'svhn':
            out_list = ['cifar10', 'imagenet_resize', 'lsun_resize']
        # MNIST Within-Dataset Experiment
        elif dataset_list[count_in] == 'mnist07':
            out_list = ['mnist89']
        # FashionMNIST Within-Dataset Experiment
        elif dataset_list[count_in] == 'fm07':
            out_list = ['fm89']
        # MNIST-FashionMNIST Between-Dataset Experiment
        elif dataset_list[count_in] == 'mnist':
            out_list = ['fm']
        # SVHN Within-Dataset Experiment
        elif dataset_list[count_in] == 'svhn07':
            out_list = ['svhn89']
            
        count_out = 0
        for results in in_list:
            print('out_distribution: ' + out_list[count_out])
            for mtype in mtypes:
                print(' {mtype:6s}'.format(mtype=mtype), end='')
            # print('\n{val:6.2f}'.format(
            #     val=100.*results['TMP']['TNR']), end='')
            print('\n{val:6.2f}'.format(
                val=100.*results['TMP']['TNR95']), end='')
            print('{val:6.2f}'.format(
                val=100.*results['TMP']['TNR99']), end='')
            
            print(' {val:6.2f}'.format(
                val=100.*results['TMP']['AUROC']), end='')
            print(' {val:6.2f}'.format(
                val=100.*results['TMP']['DTACC']), end='')
            print(' {val:6.2f}'.format(
                val=100.*results['TMP']['AUIN']), end='')
            print(' {val:6.2f}\n'.format(
                val=100.*results['TMP']['AUOUT']), end='')
            print('Input noise: ' +
                  list_best_results_index[count_in][count_out])
            print('')
            count_out += 1
        count_in += 1


if __name__ == '__main__':
    main()
