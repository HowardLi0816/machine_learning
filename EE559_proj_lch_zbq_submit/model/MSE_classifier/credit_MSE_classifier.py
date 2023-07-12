import numpy as np
from sklearn.linear_model import Ridge
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

    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--data_augment', type=str2bool, default=False)

    parser.add_argument('--ufs', type=str2bool, default=False)
    parser.add_argument('--ufs_k', type=int, default=4)
    parser.add_argument('--rfe', type=str2bool, default=True)
    parser.add_argument('--rfe_k', type=int, default=19)
    parser.add_argument('--sfs', type=str2bool, default=False)
    parser.add_argument('--sfs_k', type=int, default=2)

    parser.add_argument('--pca', type=str2bool, default=True)
    parser.add_argument('--pca_n_component', type=int, default=16)
    parser.add_argument('--lda', type=str2bool, default=False)
    parser.add_argument('--lda_n_component', type=int, default=1)

    parser.add_argument('--non_linear', type=str2bool, default=True)
    parser.add_argument('--poly', type=int, default=2)

    return parser


def load_data(args):
    '''
    get data, not use reflected data points nor perform any standardization or any data preprocessing, label to be +1/-1, no augmented
    '''
    # load datasets
    PTH = '../../data/Credit_card_datasets'
    train_data, test_data = gen_data_2(PTH, args)

    train_fea = train_data[:, :-1]
    test_fea = test_data[:, :-1]

    train_label = train_data[:, -1]
    test_label = test_data[:, -1]

    #transform labels to -1, +1
    train_label = train_label * (-2) + 1
    test_label = test_label * (-2) + 1

    train = np.concatenate([train_fea, train_label.reshape((len(train_label), 1))], axis=1)
    test = np.concatenate([test_fea, test_label.reshape((len(test_label), 1))], axis=1)

    return train, test


def MSE_binary_classifier(args):
    #generate dataset
    trainset, testset = load_data(args)
    train_data_, val_data = train_val_split(trainset)
    # print(train_data_.shape[1]-1)
    model = Ridge(alpha=1e-3)
    model.fit(train_data_[:, :-1], train_data_[:, -1])

    val_pred = model.predict(val_data[:, :-1])
    val_acc = np.sum((np.sign(val_pred) == val_data[:, -1])) / len(val_pred)
    print(f"val_acc = {val_acc:6.4f}")
    # print(f"{val_acc:6.4f}")

    y_pred_test = model.predict(testset[:, :-1])
    test_acc = np.sum((np.sign(y_pred_test) == testset[:, -1])) / len(y_pred_test)
    print(f"test_acc = {test_acc:6.4f}")
    f1 = f1_score(testset[:, -1], np.sign(y_pred_test), pos_label=1)
    print(f"F1_score = {f1:6.4f}")
    plot_cm(-1 * testset[:, -1], -1 * np.sign(y_pred_test), "../../imgs/MSE.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MSE binary classification')

    parser = init_parser(parser)

    args = parser.parse_args()
    for i in range(19, 20):
        # args.rfe_k = i
        MSE_binary_classifier(args)