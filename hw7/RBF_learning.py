import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import argparse
import matplotlib.pyplot as plt

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

    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--hidden_shape', type=int, default=3000)
    parser.add_argument('--center_method', type=str, choices=['dir', 'rand', 'k_mean'], default='dir')
    parser.add_argument('--is_cross_validation', type=str2bool, default=True)
    parser.add_argument('--split', type=int, default=4)
    parser.add_argument('--plot_data_points', type=str2bool, default=True)

    parser.add_argument('--compare_real', type=str2bool, default=False)

    return parser

def gen_data():
    DATA_PATH = './data/HW7_Pr2_datasetA/HW7_Pr2_datasetA/'
    X_train = np.load(f'{DATA_PATH}datasetA_X_train.npy')
    X_test = np.load(f'{DATA_PATH}datasetA_X_test.npy')
    y_train = np.load(f'{DATA_PATH}datasetA_y_train.npy')
    y_test = np.load(f'{DATA_PATH}datasetA_y_test.npy')
    return X_train, y_train, X_test, y_test

def RMSE(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def std_RMSE(y):
    #y is the target (y_train)
    mean = np.mean(y)
    mean_vec = np.tile(mean, len(y))
    rmse = RMSE(y, mean_vec)
    return rmse, mean

class RBF(object):
    def __init__(self, hidden_shape, gamma=1.0, center_method=None):
        #center_method choosed from ['dir', 'rand', 'k_mean']
        self.hidden_shape = hidden_shape
        self.gamma = gamma
        self.centers = None
        self.center_method = center_method
        self.linear = LinearRegression()

    def kernel_function(self, data_point, center):
        return rbf_kernel(data_point, center, self.gamma)

    def select_center_dir(self, X):
        centers = X[:self.hidden_shape]
        return centers

    def select_center_rand(self, X):
        random_args = np.random.choice(len(X), self.hidden_shape, replace=False)
        centers = X[random_args]
        return centers

    def select_center_k_mean(self, X):
        kmean = KMeans(self.hidden_shape, init='random')
        kmean.fit(X)
        centers = kmean.cluster_centers_
        return centers

    def fit(self, X, Y):
        if self.center_method == 'dir':
            self.centers = self.select_center_dir(X)
        elif self.center_method == 'rand':
            self.centers = self.select_center_rand(X)
        elif self.center_method == 'k_mean':
            self.centers = self.select_center_k_mean(X)
        rbf_out = self.kernel_function(X, self.centers)
        self.linear.fit(rbf_out, Y)

    def predict(self, X):
        rbf_out = self.kernel_function(X, self.centers)
        pred = self.linear.predict(rbf_out)
        return pred

def train(args):
    print(f'config: {args}')

    X_train, y_train, X_test, y_test = gen_data()

    rbf_model = RBF(args.hidden_shape, args.gamma, args.center_method)

    if args.is_cross_validation:
        kf = KFold(n_splits=args.split, shuffle=True)
        idx = 1
        total_rmse = []
        total_tr_rmse = []
        for train_idx, val_idx in kf.split(X_train):
            rbf_model.fit(X_train[train_idx], y_train[train_idx])
            pred = rbf_model.predict(X_train[val_idx])
            pred_rmse = RMSE(pred, y_train[val_idx])
            total_rmse.append(pred_rmse)
            pred_tr = rbf_model.predict(X_train[train_idx])
            pred_tr_rmse = RMSE(pred_tr, y_train[train_idx])
            total_tr_rmse.append(pred_tr_rmse)
            print(f'[Train{idx}]\tTrain RMSE={pred_tr_rmse:.4e}\tVal RMSE={pred_rmse:.4e}')
            idx += 1
        mean_rmse = np.mean(total_rmse)
        std_rmse = np.std(total_rmse)
        mean_rmse_tr = np.mean(total_tr_rmse)
        std_rmse_tr = np.std(total_tr_rmse)
        print(f'[Train final]\tTrain RMSE Mean={mean_rmse_tr:.4e}\tTrain RMSE Std={std_rmse_tr:.4e}\tVal RMSE Mean={mean_rmse:.4e}\tVal RMSE Std={std_rmse:.4e}')

    else:
        rbf_model.fit(X_train, y_train)
        pred = rbf_model.predict(X_test)
        pred_rmse = RMSE(pred, y_test)
        pred_tr = rbf_model.predict(X_train)
        pred_rmse_tr = RMSE(pred_tr, y_train)
        print(f'[Train final]\tTrain RMSE={pred_rmse_tr:.4e}\tTest RMSE={pred_rmse:.4e}')
        if args.plot_data_points:
            plt.figure()
            center = rbf_model.centers
            plt.ylabel("x2")
            plt.xlabel("x1")
            plt.title(f"Original data points & centers with M(K)={args.hidden_shape}, gamma={args.gamma}")
            plt.plot(center[:, 0], center[:, 1], '*', label="Centers")
            plt.plot(X_train[:, 0], X_train[:, 1], ",", color='k', label="Original train datas")
            plt.legend()
            plt.show()
        if args.compare_real:
            #plot real func

            x_x2_0_5 = np.concatenate((np.arange(0, 2.001, 0.001).reshape(-1, 1), 0.5 * np.ones(int(2.001 / 0.001)+1).reshape(-1, 1)), axis=1)
            x_x2_1_5 = np.concatenate((np.arange(0, 2.001, 0.001).reshape(-1, 1), 1.5 * np.ones(int(2.001 / 0.001)+1).reshape(-1, 1)), axis=1)
            x_x1_0_3 = np.concatenate((0.3 * np.ones(int(2.001 / 0.001)+1).reshape(-1, 1), np.arange(0, 2.001, 0.001).reshape(-1, 1)), axis=1)

            plt.figure()
            plt.ylabel("y(x1,x2)")
            plt.xlabel("x1")
            plt.title(f"y(x1,x2) vs. x1 for x2=0.5")
            plt.plot(np.arange(0, 2.001, 0.001), real_fun(x_x2_0_5), label="Real func")
            plt.legend()
            plt.show()

            plt.figure()
            plt.ylabel("y(x1,x2)")
            plt.xlabel("x1")
            plt.title(f"y(x1,x2) vs. x1 for x2=1.5")
            plt.plot(np.arange(0, 2.001, 0.001), real_fun(x_x2_1_5), label="Real func")
            plt.legend()
            plt.show()

            plt.figure()
            plt.ylabel("y(x1,x2)")
            plt.xlabel("x2")
            plt.title(f"y(x1,x2) vs. x2 for x1=0.3")
            plt.plot(np.arange(0, 2.001, 0.001), real_fun(x_x1_0_3), label="Real func")
            plt.legend()
            plt.show()

            #plot the predcted func and real func
            plt.figure()
            plt.ylabel("pred(x1,x2) / y(x1,x2)")
            plt.xlabel("x1")
            plt.title(f"pred(x1, x2) & y(x1,x2) vs. x1 for x2=0.5 (M(K)={args.hidden_shape}, gamma={args.gamma})")
            plt.plot(np.arange(0, 2.001, 0.001), real_fun(x_x2_0_5), label="Real func")
            plt.plot(np.arange(0, 2.001, 0.001), rbf_model.predict(x_x2_0_5), label="pred func")
            plt.legend()
            plt.show()

def real_fun(x):
    #x shape(n_sample, n_features)
    #return (n_sample,)
    return 10 * np.cos(np.pi / 2 * x[:, 0]) * np.sin(5 * np.pi / (1 + x[:, 0]**2)) * np.sin(np.pi * x[:, 1])

if __name__ == '__main__':
    X_train, y_train, X_test, y_test = gen_data()
    rmse, mean = std_RMSE(y_train)
    print('RMSE:', rmse)
    print('Target mean:', mean)

    parser = argparse.ArgumentParser(description='RBF training script')

    parser = init_parser(parser)

    args = parser.parse_args()

    train(args)