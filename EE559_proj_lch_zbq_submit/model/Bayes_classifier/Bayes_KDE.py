import numpy as np
from sklearn.neighbors import KernelDensity
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
    parser.add_argument('--n', default=1, type=float)
    parser.add_argument('--kernel', type=str, choices=[{'gaussian', 'tophat', 'epanechnikov',
                                                        'exponential', 'linear', 'cosine'}],
                                                        default='gaussian')

    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--data_augment', type=str2bool, default=False)

    parser.add_argument('--ufs', type=str2bool, default=False)
    parser.add_argument('--ufs_k', type=int, default=5)
    parser.add_argument('--rfe', type=str2bool, default=True)
    parser.add_argument('--rfe_k', type=int, default=5)
    parser.add_argument('--sfs', type=str2bool, default=False)
    parser.add_argument('--sfs_k', type=int, default=15)

    parser.add_argument('--pca', type=str2bool, default=False)
    parser.add_argument('--pca_n_component', type=int, default=15)
    parser.add_argument('--lda', type=str2bool, default=True)
    parser.add_argument('--lda_n_component', type=int, default=1)

    parser.add_argument('--non_linear', type=str2bool, default=True)
    parser.add_argument('--poly', type=int, default=2)

    return parser


class Bayes_Classifier:
    def __init__(self, num_class=2, allargs=None):
        self.num_class = num_class
        self.hn = np.sqrt(np.sqrt(100 / allargs.n))
        self.kde1 = KernelDensity(bandwidth=self.hn, kernel=allargs.kernel)
        self.kde2 = KernelDensity(bandwidth=self.hn, kernel=allargs.kernel)
        self.ps1 = 0
        self.ps2 = 0

    def fit(self, xdata, ydata):
        total = xdata.shape[0]
        self.ps1 = np.sum(ydata == 0) / total
        self.ps2 = np.sum(ydata == 1) / total
        self.kde1.fit(xdata[ydata == 0])
        self.kde2.fit(xdata[ydata == 1])

    def score(self, x_test, y_test):
        total = x_test.shape[0]
        y_pred = np.zeros(total)
        y_pred[np.exp(self.kde1.score_samples(x_test)) * self.ps1
               < np.exp(self.kde2.score_samples(x_test)) * self.ps2] = 1
        acc = np.sum(y_pred == y_test) / total
        return y_pred, acc


def bayes_kde(args):
    train_data, test_data = gen_data_2("../../data/Credit_card_datasets", args)
    train_data_, val_data = train_val_split(train_data)
    # print(train_data_.shape[1] - 1)
    args.n = int(train_data_.shape[0] / 10)
    model = Bayes_Classifier(allargs=args)
    model.fit(train_data_[:, :-1], train_data_[:, -1])
    y_pred_val, val_acc = model.score(val_data[:, :-1], val_data[:, -1])
    print(f"val_acc = {val_acc:6.4f}")
    # print(f"{val_acc:6.4f}")
    y_pred_test, test_acc = model.score(test_data[:, :-1], test_data[:, -1])
    print(f"test_acc = {test_acc:6.4f}")
    f1 = f1_score(test_data[:, -1], y_pred_test, pos_label=0)
    print(f"F1_score = {f1:6.4f}")
    plot_cm(test_data[:, -1], y_pred_test, "../../imgs/bayes_kde_RFE8.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Bayes_KDE')

    parser = init_parser(parser)

    args = parser.parse_args()

    for i in range(4, 5):
        # args.rfe_k = i
        bayes_kde(args)
