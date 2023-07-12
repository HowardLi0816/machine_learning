import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.linear_model import Perceptron
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from utils.plotDecBoundaries import plotDecBoundaries

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

    parser.add_argument('--standardize', type=str2bool, default=True)
    parser.add_argument('--cross_val_fold', type=int, default=20)
    parser.add_argument('--fea_include', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--plot_bry', type=str2bool, default=False)
    parser.add_argument('--shuffle', type=str2bool, default=True)
    parser.add_argument('--is_cross_validation', type=str2bool, default=True)
    parser.add_argument('--run_times', type=int, default=5)
    parser.add_argument('--dim_reduce_method', type=str, choices=['artifical', 'PCA', 'MDA'], default='artifical')

    return parser

def gen_data():
    data_path = "./data/wine_data.csv"
    data = pd.read_csv(data_path, header=None).to_numpy()
    labels, features = data[:, 0], data[:, 1:]
    return features, labels


def train_perceptron(args):
    fea, labels = gen_data()
    if args.standardize:
        std = StandardScaler()
        std.fit(fea)
        fea = std.transform(fea)

    if args.dim_reduce_method == 'artifical':
        include_features = fea[:, (args.fea_include[0], args.fea_include[1])]
    elif args.dim_reduce_method == 'PCA':
        PCA_model = PCA(n_components=2)
        include_features = PCA_model.fit_transform(fea)
    elif args.dim_reduce_method == 'MDA':
        MDA_model = LDA(n_components=2)
        include_features = MDA_model.fit_transform(fea, labels)

    plotDecBoundaries(include_features, labels, fsize=(6, 6), title=f"Data gen by {args.dim_reduce_method}, standardize={args.standardize}")

    models = []
    err = []

    for i in range(args.run_times):
        print(f"###########################{i}th experiment###########################")
        model = Perceptron()
        if args.is_cross_validation:
            kf = KFold(n_splits=args.cross_val_fold, shuffle=args.shuffle)
            idx = 1
            total_val_err = []
            for train_idx, val_idx in kf.split(include_features):
                model.fit(include_features[train_idx], labels[train_idx])
                val_err = 1 - model.score(include_features[val_idx], labels[val_idx])
                total_val_err.append(val_err)
                print(f'[Train{idx}]\tVal Err={val_err:.4e}')
                if (idx == 1):
                    model_tmp = model
                    models.append(model_tmp)
                idx += 1
            mean_val_err = np.mean(total_val_err)
            print(
                f'[Train final]\tVal Err Mean={mean_val_err:.4e}')
            err.append(mean_val_err)

    print(f"For the {args.run_times} runs, the error rate are:", err)
    err_mean = np.mean(err)
    err_std = np.std(err)
    print(f"The mean is {err_mean} and the standard derivation is {err_std}")

    lowest_err_idx = np.argmin(err)
    model_low = models[lowest_err_idx]
    plotDecBoundaries(include_features, labels, func=model_low.predict, plot_bry=args.plot_bry, fsize=(6, 6), title=f"The lowest error rate for standardize={args.standardize}, {args.dim_reduce_method}")

    highest_err_idx = np.argmax(err)
    model_high = models[highest_err_idx]
    plotDecBoundaries(include_features, labels, func=model_high.predict, plot_bry=args.plot_bry, fsize=(6, 6), title=f"The highest error rate for standardize={args.standardize}, {args.dim_reduce_method}")



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP training script')

    parser=init_parser(parser)

    args = parser.parse_args()

    train_perceptron(args)
