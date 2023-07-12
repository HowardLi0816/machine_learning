import numpy as np
import matplotlib.pyplot as plt
from plotDecBoundaries import plotDecBoundaries
from draw_data import draw_data
from sklearn.neighbors import KernelDensity

def decision_func(x):
    mean1 = np.array([-4, 0])
    var1 = np.array([[4, 0], [0, 1]])
    mean2 = np.array([-0.5, 0])
    var2 = np.array([[0.16, 0], [0, 9]])
    x1_jud = np.logical_and(x[:, 0] >= np.ones(len(x)) * 0, x[:, 0] <= np.ones(len(x)) * 2)
    x2_jud = np.logical_and(x[:, 1] >= np.ones(len(x)) * -1, x[:, 1] <= np.ones(len(x)) * 1)

    left = (0.35 / np.pi) / np.sqrt(np.linalg.det(var1)) * np.exp(-0.5 * np.sum((x - mean1).dot(np.linalg.inv(var1)) * (x-mean1), axis=1)) + 0.075 * x1_jud * x2_jud
    right = (0.5 / np.pi) / np.sqrt(np.linalg.det(var2)) * np.exp(-0.5 * np.sum((x - mean2).dot(np.linalg.inv(var2)) * (x-mean2), axis=1))


    dec = ((left - right) < 0) + 1
    return dec


def main_fun(n):
    mean = np.array([[-4, 0], [1, 0], [-0.5, 0]])
    plotDecBoundaries(mean, func=decision_func, title="Decision Boundary of True Bayes minimum error classifier")

    train_data = draw_data(20000)
    plotDecBoundaries(mean, func=decision_func, title="First 2000 data points",
                      plt_mean=False, plt_dec=False, plt_data=True, training=train_data[:2000, :-1], label_train=train_data[:2000, -1])

    test_data = draw_data(10000)

    pred_label = decision_func(test_data[:, :-1])
    acc = np.sum(pred_label == test_data[:, -1]) / len(test_data)
    print("Bayes minimum error classifier acc:", acc)

    train = train_data[:n, :]
    ps1 = np.sum((train[:, -1] - 2) * (-1)) / len(train)
    ps2 = np.sum(train[:, -1] - 1) / len(train)

    print("P_h(S1) = ", ps1)
    print("P_h(S2) = ", ps2)

    bandwidth = (100/n)**(1/4)
    train_fea = train[:, :-1]
    label = train[:, -1]
    s1_data = train_fea[label==1]
    s2_data = train_fea[label==2]
    kde1 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(s1_data)
    kde2 = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(s2_data)

    pred1 = np.exp(kde1.score_samples(test_data[:, :-1])) * ps1
    pred2 = np.exp(kde2.score_samples(test_data[:, :-1])) * ps2

    def gen_dec(x):
        p1 = np.exp(kde1.score_samples(x)) * ps1
        p2 = np.exp(kde2.score_samples(x)) * ps2
        return (p1 < p2) + 1

    pred = (pred1 < pred2) + 1

    acc_kde = np.sum(pred == test_data[:, -1]) / len(test_data)
    print(f"KDE with n={n}, acc = ", acc_kde)
    plotDecBoundaries(mean, func=gen_dec, title=f"Decision Boundary of KDE Bayes minimum error classifier with n={n}", plt_mean=False)


if __name__ == '__main__':
    main_fun(20000)
