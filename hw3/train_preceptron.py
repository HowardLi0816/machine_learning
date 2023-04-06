import numpy as np
import matplotlib.pyplot as plt
from utils.dataset import data_gen, data_shuffle, reflect_data
from utils.plotDecBoundaries import plotDecBoundaries
import argparse
import math
from sklearn.preprocessing import normalize

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
                        choices=['h1d1', 'h1d2', 'h1d3', 'h3'],
                        default='h1d1')
    parser.add_argument('--max_iter', default=1e4, type=int)
    parser.add_argument('--lr', default=1, type=float)
    parser.add_argument('--adjust_lr', default=False, type=str2bool)
    parser.add_argument('--w_init_para', default=0.1, type=float)
    parser.add_argument('--shuffle', default=True, type=str2bool)
    parser.add_argument('--reflect_data', default=True, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['BSGD', 'SGDV1'], default='BSGD')
    parser.add_argument('--plot_learning_curve', default=False, type=str2bool)
    parser.add_argument('--plot_dec_boundary', default=False, type=str2bool)
    parser.add_argument('--normalize', default=False, type=str2bool)
    parser.add_argument('--plot_dis_hist', default=False, type=str2bool)
    parser.add_argument('--activate_func', type=str, choices=['Relu', 'Softmax'], default='Relu')

    return parser

def init_w(fea_num, args):
    w = np.ones(fea_num)
    w = w * args.w_init_para
    return w

def criterion(w, ref_data, args):
    data_num = ref_data.shape[0]
    fea_map = ref_data[:, :-1]
    j_w = 0
    misclass = 0
    #print("********")
    for i in range(data_num):
        g_x = np.dot(w, fea_map[i, :])
        #print(g_x)
        if args.activate_func == 'Relu':
            if g_x <= 0:
                j_w = j_w + (g_x * -1)
        elif args.activate_func == 'Softmax':
            j_w = j_w + np.log(1 + np.exp(-1 * g_x))

    return j_w

def validate(w, non_refl_data):
    data_num = non_refl_data.shape[0]
    fea_map = non_refl_data[:, :-1]
    labels = non_refl_data[:, -1]
    mis_cls = 0
    for i in range(data_num):
        g_x = np.dot(w, fea_map[i, :])
        if g_x > 0:
            pred_labels = 1
        elif g_x < 0:
            pred_labels = 2
        elif g_x == 0:
            pred_labels = -1
        if pred_labels != labels[i]:
            mis_cls += 1
    err = mis_cls/data_num
    return err


def train_per(args):

    #load the training data
    print('Loading training data and data processing')
    read_data_from = args.dataset
    if read_data_from == 'h1d1':
        data_path = './HW1_datasets/dataset1_train.csv'
        data, fea, labels, labels_type = data_gen(data_path, 'hw1')
        test_path = './HW1_datasets/dataset1_test.csv'
        test, test_fea, test_labels, test_labels_type = data_gen(test_path, 'hw1')
    elif read_data_from == 'h1d2':
        data_path = './HW1_datasets/dataset2_train.csv'
        data, fea, labels, labels_type = data_gen(data_path, 'hw1')
        test_path = './HW1_datasets/dataset2_test.csv'
        test, test_fea, test_labels, test_labels_type = data_gen(test_path, 'hw1')
    elif read_data_from == 'h1d3':
        data_path = './HW1_datasets/dataset3_train.csv'
        data, fea, labels, labels_type = data_gen(data_path, 'hw1')
        test_path = './HW1_datasets/dataset3_test.csv'
        test, test_fea, test_labels, test_labels_type = data_gen(test_path, 'hw1')
    elif read_data_from == 'h3':
        data_path = './HW3_datasets/breast_cancer_train.npy'
        data, fea, labels, labels_type = data_gen(data_path, 'hw3')
        test_path = './HW3_datasets/breast_cancer_test.npy'
        test, test_fea, test_labels, test_labels_type = data_gen(test_path, 'hw3')

    if args.normalize:
        train_num = data.shape[0]
        nor_data = 100 * normalize(np.concatenate((fea, test_fea), axis=0), axis=0, norm='l1')
        fea = nor_data[:train_num]
        test_fea = nor_data[train_num:]
        data = np.concatenate((fea, data[:, -1].reshape((train_num, 1))), axis=1)
        test = np.concatenate((test_fea, test[:, -1].reshape((test.shape[0], 1))), axis=1)

    if args.plot_dis_hist:
        cls_data = []
        for i in range(labels_type.shape[0]):
            cls_data.append([])
        for n in range(data.shape[0]):
            label = data[n, -1]
            cls_data[int(label)-1].append(data[n])

    #print(data)
    #data preprocessing
    norm = np.ones((data.shape[0], 1))
    norm_test = np.ones((test.shape[0], 1))
    orig_data = np.concatenate((norm, data), axis=1)
    train_val = orig_data
    fea = np.concatenate((norm, fea), axis=1)
    test_data = np.concatenate((norm_test, test), axis=1)
    if args.shuffle == True:
        orig_data = data_shuffle(orig_data)
    if args.reflect_data == True:
        orig_data = reflect_data(orig_data, labels_type)


    #trining param
    lr = args.lr
    max_iter = args.max_iter
    w = init_w(fea.shape[1], args)
    N = orig_data.shape[0]

    #start training
    print("Start training")
    iter = 0
    epoch = 1
    n = 0
    best_loss = math.inf
    best_w = np.zeros(fea.shape[1])
    best_epoch = 0
    best_iter = 0
    best_n = 0
    last_epoch = 1
    loss_list = []
    err_list = []
    while iter < max_iter:
        if args.train_method == 'SGDV1':
            if epoch > last_epoch:
                print('############SGDV1 shuffle data, epoch:', epoch)
                last_epoch = epoch
                orig_data = data_shuffle(orig_data)

        da = orig_data[n, :-1]

        #calculate the loss and err for the new w
        loss = criterion(w, orig_data, args)
        err = validate(w, train_val)

        if args.plot_learning_curve:
            loss_list.append(loss)
            err_list.append(err)

        #test_loss, test_err = criterion(w, test_data)
        test_err = validate(w, test_data)
        #print(w)
        print("epoch:", epoch, "iter:", iter, "n:", n, "------loss:", loss, "acc:", 1-err, "test_acc:", 1-test_err)
        if loss == 0:
            best_w = w
            best_epoch = epoch
            best_iter = iter
            best_n = n
            print("Training converges---err rate:", err)
            print("Data is linearly separable")
            break
        if loss < best_loss and iter != 0:
            best_loss = loss
            best_w = w
            best_epoch = epoch
            best_iter = iter
            best_n = n
        #update the w for next iteration
        g_x = np.dot(w, da)
        if args.activate_func == 'Relu':
            if g_x <= 0:
                w = w + lr * da
            else:
                w = w
        elif args.activate_func == 'Softmax':
            w = w + lr * da * (np.exp(-1 * g_x) / (1 + np.exp(-1 * g_x)))


        #update the idx
        iter += 1
        n += 1
        if n == N:
            n = 0
            epoch += 1

    final_w = best_w
    final_loss = criterion(final_w, orig_data, args)
    final_err = validate(final_w, train_val)
    print("Train end, best appears at ---epoch:", best_epoch, "iter:", best_iter, "n:", best_n, "---final loss:", final_loss, "final acc:", 1-final_err)
    final_test_err = validate(final_w, test_data)
    print("Final test ---final acc:", 1-final_test_err)

    if args.plot_learning_curve:
        ep = int(iter / N) + 1
        loss = np.array(loss_list)
        err = np.array(err_list)
        if ep <= 10:
            x = np.arange(0,iter+1)
        else:
            x = np.arange(1, ep+1)
            crit = np.zeros(ep)
            error = np.zeros(ep)
            for e in range(ep):
                if e==(ep-1):
                    loss_ep = loss[(e*N):]
                    err_ep = err[(e*N):]
                else:
                    loss_ep = loss[(e * N): (e + 1)*N]
                    err_ep = err[(e * N): (e + 1)*N]
                crit[e] = np.mean(loss_ep)
                error[e] = np.mean(err_ep)
            loss = crit
            err = error
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(x, loss, label="J(w)", color='b')
        ax1.set_ylabel("J(Loss)")
        ax1.set_title("J(w)<blue> / error rate<red> vs epochs / iterations")
        ax2 = ax1.twinx()
        ax2.plot(x, err, label="error rate", color='r')
        ax2.set_ylabel("error rate")
        plt.show()

    if args.plot_dec_boundary:
        plotDecBoundaries(fea[:, 1:], labels, final_w)

    if args.plot_dis_hist:
        plt.figure()
        plt.title('Dis Hist of different classes')
        plt.xlabel('distance')
        colors = ['r', 'b', 'g', 'y']
        for i in range(len(cls_data)):
            dis = []
            cls = np.array(cls_data[i])
            cls = cls[:, :-1]
            cls = np.concatenate((np.ones((cls.shape[0], 1)), cls), axis=1)
            for j in range(cls.shape[0]):
                d = np.dot(final_w, cls[j]) / np.linalg.norm(final_w)
                dis.append(d)
            dis = np.array(dis)
            name = 'class ' + str(i)
            plt.hist(dis, bins=100, label=name, color=colors[i], alpha=0.5)
        plt.show()
    return final_w


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Perceptron training')

    parser = init_parser(parser)

    args = parser.parse_args()

    final_w = train_per(args)

    print("final_w:", final_w)