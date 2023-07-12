import numpy as np
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
    parser.add_argument('--ufs_k', type=int, default=3)
    parser.add_argument('--rfe', type=str2bool, default=False)
    parser.add_argument('--rfe_k', type=int, default=2)
    parser.add_argument('--sfs', type=str2bool, default=True)
    parser.add_argument('--sfs_k', type=int, default=7)

    parser.add_argument('--pca', type=str2bool, default=False)
    parser.add_argument('--pca_n_component', type=int, default=12)
    parser.add_argument('--lda', type=str2bool, default=False)
    parser.add_argument('--lda_n_component', type=int, default=1)

    parser.add_argument('--non_linear', type=str2bool, default=False)
    parser.add_argument('--poly', type=int, default=2)

    return parser


class Baseline:
    def __init__(self, num_feature=23, num_class=2, allargs=None):
        self.num_class = num_class
        self.means = np.zeros([num_class, num_feature])

    def fit(self, xdata, ydata):
        for i in range(self.num_class):
            self.means[i, :] = np.mean(xdata[ydata == i, :], axis=0)

    def score(self, x_test, y_test):
        total = x_test.shape[0]
        dist = np.zeros([total, self.num_class])
        for i in range(self.num_class):
            dist[:, i] = np.linalg.norm(x_test - self.means[i, :], ord=2, axis=1)
        y_pred = np.argmin(dist, axis=1)
        acc = np.sum(y_pred == y_test) / total
        return y_pred, acc

def baseline_system(args):
    train_data, test_data = gen_data_2("../../data/Credit_card_datasets", args)
    train_data_, val_data = train_val_split(train_data)
    # print(train_data_.shape[1]-1)
    triv = Baseline(num_feature=train_data_.shape[1]-1)
    triv.fit(train_data_[:, :-1], train_data_[:, -1])
    y_pred, val_acc = triv.score(val_data[:, :-1], val_data[:, -1])
    print(f"val_acc = {val_acc:6.4f}")
    # print(f"{val_acc:6.4f}")
    y_pred_test, test_acc = triv.score(test_data[:, :-1], test_data[:, -1])
    print(f"test_acc = {test_acc:6.4f}")
    f1 = f1_score(test_data[:, -1], y_pred_test, pos_label=0)
    print(f"F1_score = {f1:6.4f}")
    plot_cm(test_data[:, -1], y_pred_test, "../../imgs/baseline.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Baseline')

    parser = init_parser(parser)

    args = parser.parse_args()

    for i in range(1, 2):
        # args.pca_n_component = i
        baseline_system(args)
