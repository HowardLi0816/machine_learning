import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils.PlotNonlinear import plotDecBoundaries_Nonlinear
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
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
                        choices=['hw5'],
                        default='hw5')
    parser.add_argument('--max_iter', default=10000, type=int)
    parser.add_argument('--reg_type', type=str, choices=['l1', 'l2'], default='l2')
    parser.add_argument('--nonlinear_list', nargs='+', type=int, default=[1])
    parser.add_argument('--l2_param_list', nargs='+', type=float, default=[1])
    parser.add_argument('--fea_num', type=int, default=2)
    parser.add_argument('--plot_dec_boundary', type=str2bool, default=True)
    parser.add_argument('--plot_acc_p', type=str2bool, default=True)
    parser.add_argument('--plot_jmse_p', type=str2bool, default=True)
    parser.add_argument('--plot_teacc_loglam', type=str2bool, default=True)
    parser.add_argument('--get_weights', type=str2bool, default=True)
    parser.add_argument('--out_weight_lam', type=float, default=1)
    parser.add_argument('--out_weight_p', type=int, default=1)
    parser.add_argument('--teacc_loglam_p_idx', nargs='+', type=int, default=[1])

    return parser

def gen_data(args):
    '''
    get data, not use reflected data points nor perform any standardization or any data preprocessing, label to be +1/-1, no augmented
    '''
    train_path = f'./data/{args.dataset}_train.csv'
    test_path = f'./data/{args.dataset}_test.csv'

    train_data = pd.read_csv(train_path, header=None).to_numpy()
    test_data = pd.read_csv(test_path, header=None).to_numpy()

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

def cal_mse(a, b, w, lam):
    return np.sum((a-b)**2) / len(a) + lam * np.sum(w**2)

def MSE_binary_classifier(args):
    #generate dataset
    trainset, testset = gen_data(args)
    train_fea_org, train_label = trainset[:, :-1], trainset[:, -1]
    test_fea_org, test_label = testset[:, :-1], testset[:, -1]

    '''
    create np.array to store train accuracy and test accuracy
    create np.array to store train Jmse
    shape: (len(lambda), len(p))
    '''
    lam_list = args.l2_param_list
    p_list = args.nonlinear_list
    tr_acc = np.zeros((len(lam_list), len(p_list)))
    te_acc = np.zeros((len(lam_list), len(p_list)))
    Jmse = np.zeros((len(lam_list), len(p_list)))
    idx1 = 0
    idx2 = 0

    for l in lam_list:
        for p in p_list:
            # generate nonlinear data
            poly = PolynomialFeatures(degree=p, include_bias=True)
            train_fea = poly.fit_transform(train_fea_org)
            test_fea = poly.fit_transform(test_fea_org)

            #create model and train
            model = Ridge(alpha=l, fit_intercept=False)
            model.fit(train_fea, train_label)

            if args.out_weight_lam == l and args.out_weight_p == p:
                print(f'Weight of model lam={l:.1f} p={p}:')
                print(model.coef_)

            #predict and acc
            train_pred = model.predict(train_fea)
            test_pred = model.predict(test_fea)
            #print(test_label)
            train_acc = np.sum((np.sign(train_pred) == train_label)) / len(train_fea)
            test_acc = np.sum((np.sign(test_pred) == test_label)) / len(test_fea)
            print(f'[Train] \t [lam={l:.1f}] \t [p={p}] \t Acc {train_acc:6.4f}')
            print(f'[Test] \t [lam={l:.1f}] \t [p={p}] \t Acc {test_acc:6.4f}')
            tr_acc[idx1, idx2] = train_acc
            te_acc[idx1, idx2] = test_acc

            #cal mse
            Jmse[idx1, idx2] = cal_mse(train_pred, train_label, model.coef_, l)

            if not os.path.exists(f"./fig/lam_{l:.1f}"):
                os.makedirs(f"./fig/lam_{l:.1f}")


            if args.plot_dec_boundary:
                save_path = f"./fig/lam_{l:.1f}/p={p}_dec.png"
                plotDecBoundaries_Nonlinear(train_fea_org, train_label, poly.fit_transform, model.predict, save_path, fsize=(6,4), legend_on = True)

            idx2 += 1

        if args.plot_acc_p:
            plt.figure()
            plt.plot(np.array(p_list), tr_acc[idx1, :], 'o', color='r', label='train_acc')
            plt.plot(np.array(p_list), te_acc[idx1, :], '*', color='b', label='test_acc')
            plt.plot(np.array(p_list), tr_acc[idx1, :], color='r')
            plt.plot(np.array(p_list), te_acc[idx1, :], color='b')
            plt.xlabel('p value')
            plt.ylabel('acc')
            plt.title(f"train and test acc vs p for lam={l:.1f}")
            plt.legend()
            plt.savefig(f"./fig/lam_{l:.1f}/acc_vs_p.png")
            plt.show()

        if args.plot_jmse_p:
            plt.figure()
            plt.plot(np.array(p_list), Jmse[idx1, :], 'o', color='r', label='train_Jmse')
            plt.plot(np.array(p_list), Jmse[idx1, :], color='r')
            plt.xlabel('p value')
            plt.ylabel('train_Jmse')
            plt.title(f"train Jmse vs p for lam={l:.1f}")
            plt.legend()
            plt.savefig(f"./fig/lam_{l:.1f}/Jmse_vs_p.png")
            plt.show()

        idx1 += 1
        idx2 = 0

    if args.plot_teacc_loglam:
        if not os.path.exists(f"./fig/te_acc_loglam"):
            os.makedirs(f"./fig/te_acc_loglam")
        plt.figure()
        col = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        sign = ['o', '*', '.', 'p', '+', 'h', 'd', 's']
        plot_p = []
        for idx in args.teacc_loglam_p_idx:
            plot_p.append(p_list[idx])
            for j in range(len(lam_list)):
                if lam_list[j] == 0:
                    lam_list[j] = 0.1
            plt.plot(np.log10(np.array(lam_list)), te_acc[:, idx], sign[idx], color=col[idx], label=f'p={p_list[idx]}')
            plt.plot(np.log10(np.array(lam_list)), te_acc[:, idx], color=col[idx])
        plt.xlabel('log(lam) value')
        plt.ylabel('test acc')
        plt.title(f"test acc vs log(lam) for p={plot_p}")
        plt.legend()
        plt.savefig(f"./fig/te_acc_loglam/te_acc_loglam_p.png")
        plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MSE binary classification')

    parser = init_parser(parser)

    args = parser.parse_args()

    MSE_binary_classifier(args)