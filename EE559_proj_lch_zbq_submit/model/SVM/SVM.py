import numpy as np
from sklearn.svm import SVC
import argparse
import sys
from sklearn.metrics import f1_score
# sys.path.append('G:\python\EE_code\\559\project')
sys.path.append('E:\Python\EE559\EE559_proj_lch_zbq')
from utils.gen_data import gen_data_2
from utils.gen_data import train_val_split
from utils.performance import plot_cm


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
    parser.add_argument('--C', default=1, type=float)
    parser.add_argument('--kernel', type=str, choices=['linear', 'poly', 'rbf', 'sigmoid', 'precomuted'], default='rbf')
    parser.add_argument('--gamma', type=float, default=10)

    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--data_augment', type=str2bool, default=False)

    parser.add_argument('--ufs', type=str2bool, default=False)
    parser.add_argument('--ufs_k', type=int, default=4)
    parser.add_argument('--rfe', type=str2bool, default=False)
    parser.add_argument('--rfe_k', type=int, default=5)
    parser.add_argument('--sfs', type=str2bool, default=True)
    parser.add_argument('--sfs_k', type=int, default=2)

    parser.add_argument('--pca', type=str2bool, default=False)
    parser.add_argument('--pca_n_component', type=int, default=15)
    parser.add_argument('--lda', type=str2bool, default=True)
    parser.add_argument('--lda_n_component', type=int, default=1)

    parser.add_argument('--non_linear', type=str2bool, default=True)
    parser.add_argument('--poly', type=int, default=1)

    return parser


def SVM(args):
    train_data, test_data = gen_data_2("../../data/Credit_card_datasets", args)
    train_data_, val_data = train_val_split(train_data)
    model = SVC(C=args.C, kernel=args.kernel, gamma=args.gamma)
    # print(train_data_.shape[1]-1)
    model.fit(train_data_[:, :-1], train_data_[:, -1])
    val_acc = model.score(val_data[:, :-1], val_data[:, -1])
    print(f"val_acc = {val_acc:6.4f}")
    # print(f"{val_acc:6.4f}")
    test_acc = model.score(test_data[:, :-1], test_data[:, -1])
    y_pred_test = model.predict(test_data[:, :-1])
    print(f"test_acc = {test_acc:6.4f}")
    f1 = f1_score(test_data[:, -1], y_pred_test, pos_label=0)
    print(f"F1_score = {f1:6.4f}")
    plot_cm(test_data[:, -1], y_pred_test, "../../imgs/SVM_ufs4.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SVM')

    parser = init_parser(parser)

    args = parser.parse_args()
    for i in range(6, 7):
        # args.rfe_k = i
        SVM(args)

