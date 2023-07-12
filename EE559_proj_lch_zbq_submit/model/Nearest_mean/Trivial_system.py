import numpy as np
import argparse
import sys
from sklearn.metrics import f1_score
# sys.path.append('G:\python\EE_code\\559\project')
sys.path.append('E:\Python\EE559\EE559_proj_lch_zbq')
from utils.gen_data import gen_data_2
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
    parser.add_argument('--ufs_k', type=int, default=15)
    parser.add_argument('--rfe', type=str2bool, default=False)
    parser.add_argument('--rfe_k', type=int, default=15)
    parser.add_argument('--sfs', type=str2bool, default=False)
    parser.add_argument('--sfs_k', type=int, default=15)

    parser.add_argument('--pca', type=str2bool, default=False)
    parser.add_argument('--pca_n_component', type=int, default=15)
    parser.add_argument('--lda', type=str2bool, default=False)
    parser.add_argument('--lda_n_component', type=int, default=1)

    parser.add_argument('--non_linear', type=str2bool, default=False)
    parser.add_argument('--poly', type=int, default=1)

    return parser


class Trivial:
    def __init__(self, num_class=2, allargs=None):
        self.num_class = num_class
        self.prob = np.zeros(num_class + 1)

    def fit(self, ydata):
        self.prob[0] = 0
        total = ydata.shape[0]
        for i in range(self.num_class):
            self.prob[i + 1] = np.sum(ydata == i) / total + self.prob[i]
        if self.prob[-1] != 1:
            self.prob[-1] = 1

    def score(self, y_test):
        total = y_test.shape[0]
        y_gene = np.random.random(total)
        y_pred = np.zeros(total)
        for i in range(self.num_class):
            y_pred[(y_gene >= self.prob[i]) * (y_gene < self.prob[i + 1])] = i
        acc = np.sum(y_pred == y_test) / total
        return y_pred, acc


def trivial_system(args):
    train_data, test_data = gen_data_2("../../data/Credit_card_datasets", args)
    triv = Trivial()
    triv.fit(train_data[:, -1])
    y_pred, test_acc = triv.score(test_data[:, -1])
    print(f"test_acc = {test_acc:6.4f}")
    f1 = f1_score(test_data[:, -1], y_pred, pos_label=0)
    print(f"F1_score = {f1:6.4f}")
    plot_cm(test_data[:, -1], y_pred, "../../imgs/trivial.png")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trivial')

    parser = init_parser(parser)

    args = parser.parse_args()

    trivial_system(args)




