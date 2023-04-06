import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from plotSVMBoundaries import plotSVMBoundaries
import argparse
import os

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_parser(parser):
    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['dataset1', 'dataset3'],
                        default='dataset1')
    parser.add_argument('--C', default=1, type=float)
    parser.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomuted'], default='rbf')
    parser.add_argument('--gamma', type=float, default=1)
    parser.add_argument('--show_support_vector', type=str2bool, default=False)
    parser.add_argument('--plot_dec_boundary', type=str2bool, default=True)
    #parser.add_argument('--give_linear_w', type=str2bool, default=True)

    return parser

def gen_data(args):
    '''
    get data, not use reflected data points nor perform any standardization or any data preprocessing, no augmented
    '''
    train_path = f'./data/{args.dataset}_train.csv'
    test_path = f'./data/{args.dataset}_test.csv'

    train_data = pd.read_csv(train_path, header=None).to_numpy()
    test_data = pd.read_csv(test_path, header=None).to_numpy()

    return train_data, test_data

def SVM(args):
    train, test = gen_data(args)
    model = SVC(C=args.C, kernel=args.kernel, gamma=args.gamma)
    model.fit(train[:, :-1], train[:, -1])
    train_acc = model.score(train[:, :-1], train[:, -1])
    test_acc = model.score(test[:, :-1], test[:, -1])
    print(f"train_acc = {train_acc:6.4f}")
    print(f"test_acc = {test_acc:6.4f}")

    if args.kernel == 'linear':
        weights = model.coef_
        print(weights)
        w0 = model.intercept_
        print(w0)

    if args.plot_dec_boundary:
        sup_vec = []
        if args.show_support_vector:
            sup_vec = model.support_vectors_
        plotSVMBoundaries(train[:, :-1], train[:, -1], model, sup_vec)
        plotSVMBoundaries(test[:, :-1], test[:, -1], model)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SVM')

    parser = init_parser(parser)

    args = parser.parse_args()

    final_w = SVM(args)
