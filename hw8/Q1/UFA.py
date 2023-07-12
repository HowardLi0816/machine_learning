import numpy as np
import matplotlib.pyplot as plt
import argparse

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

    parser.add_argument('--M', type=int, default=2)
    parser.add_argument('--domain', nargs='+', type=int, default=[0, 1])

    return parser

def original_func(x):
    return np.exp(-2 * x) * np.cos(4 * np.pi * x)

def plot_func(args):
    x_pred = np.linspace(args.domain[0], args.domain[1], args.M)
    x_gt = np.linspace(args.domain[0], args.domain[1], 10000)
    y_pred = original_func(x_pred)
    y_gt = original_func(x_gt)

    plt.figure()
    plt.ylabel("y")
    plt.xlabel("x")
    plt.title(f"Original and predicted function with M={args.M}")
    plt.plot(x_pred, y_pred, label=f"predicted function")
    plt.plot(x_gt, y_gt, label=f"original function")
    plt.legend()
    plt.show()

#Copy from Prof Chugg's github: https://github.com/keithchugg/ee559_spring2023/blob/main/hw_helpers/nmse_01.py
def normalized_mse_01(f, f_hat, x_grid, G=10000):
    # f: target function
    # f_hat: values of f_hat on the grid x_grid on [0,1]
    # x_grid a "coarse" grid on [0,1].  This has M point from the approximation.
    # G: grid size for a fine grid used to approximate the integral.

    x_fine =  np.linspace(0, 1, G)                  # create the fine grid
    f_fine = f(x_fine)                              # evaluate f on the fine grid
    f_hat_fine = np.interp(x_fine, x_grid, f_hat)   # interpolate f_hat to the fine grid
    sq_error = (f_fine - f_hat_fine) ** 2           # compute squared error
    mse = np.mean(sq_error)                         # this is a scalar multiple of the integral (approximately)
    ref = np.mean(f_fine ** 2)                      # Energy in target; off by same scalar as mse
    return mse / ref                                # scalar values cancel

def plot_NMSE(args):
    M_list = []
    NMSE_list = []
    NMSE = np.inf
    M = 2
    while(NMSE >= -40):
        M_list.append(M)
        x_grid = np.linspace(args.domain[0], args.domain[1], M)
        f_hat = original_func(x_grid)
        NMSE = 10 * np.log10(normalized_mse_01(original_func, f_hat, x_grid))
        NMSE_list.append(NMSE)
        #print("M:", M, "NMSE:", NMSE)
        M += 1

    plt.figure()
    plt.ylabel("10log10(NMSE), unit: dB")
    plt.xlabel("M")
    plt.title(f"M vs. 10log10(NMSE)")
    plt.plot(M_list, NMSE_list, label=f"10log10(NMSE)")
    plt.legend()
    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='MLP training script')

    parser=init_parser(parser)

    args = parser.parse_args()

    plot_func(args)

    plot_NMSE(args)