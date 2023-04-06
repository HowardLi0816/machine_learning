import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import h5py
from utils.PlotNonlinear import plotDecBoundaries_Nonlinear

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
                        choices=['dataset1', 'dataset2', 'dataset3'],
                        default='dataset1')
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--lr', default=1, type=float)
    #parser.add_argument('--min_lr', default=1e-3, type=float)
    #parser.add_argument('--warmup', default=10, type=int, help='number of warmup epochs')
    #parser.add_argument('--adjust_lr', default=False, type=str2bool)
    parser.add_argument('--shuffle', default=True, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['SGD'], default='SGD')
    # parser.add_argument('--binary_class', type=int, default=2)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--reflect_data', type=str2bool, default=True)
    parser.add_argument('--plot_dec_boundary', type=str2bool, default=True)
    parser.add_argument('--nonlinear', type=int, default=1)
    parser.add_argument('--fea_num', type=int, default=2)
    #parser.add_argument('--print_freq', type=int, default=50)

    return parser

def gen_data(args):
    '''
    output is augmented and normalized trian and test data with labels in the last column, unreflected data
    '''
    train_path = f'./data/{args.dataset}_train.csv'
    test_path = f'./data/{args.dataset}_test.csv'

    train_data = pd.read_csv(train_path, header=None).to_numpy()
    test_data = pd.read_csv(test_path, header=None).to_numpy()

    train_fea = train_data[:, :-1]
    test_fea = test_data[:, :-1]
    fea_num = train_fea.shape[1]

    train_label = train_data[:, -1]
    test_label = test_data[:, -1]

    num_train = len(train_fea)
    num_test = len(test_fea)

    le = LabelEncoder()
    le.fit(train_label)

    num_class = len(le.classes_)
    train_label_rep = le.transform(train_label)
    test_label_rep = le.transform(test_label)

    #normalize the data
    if args.normalize:
        scalar = StandardScaler()
        scalar.fit(train_fea)
        train_fea = scalar.transform(train_fea)
        test_fea = scalar.transform(test_fea)

    if args.nonlinear >= 2:
        for i in range(fea_num):
            for j in np.arange(i, fea_num):
                train_fea = np.concatenate((train_fea, (train_fea[:, i] * train_fea[:, j]).reshape((len(train_fea), 1))), axis=1)
                test_fea = np.concatenate((test_fea, (test_fea[:, i] * test_fea[:, j]).reshape((len(test_fea), 1))), axis=1)

    elif args.nonlinear == 3:
        for i in range(fea_num):
            for j in np.arange(i, fea_num):
                for k in np.arange(j, fea_num):
                    train_fea = np.concatenate(
                        (train_fea, (train_fea[:, i] * train_fea[:, j] * train_fea[:, k]).reshape((len(train_fea), 1))), axis=1)
                    test_fea = np.concatenate((test_fea, (test_fea[:, i] * test_fea[:, j] * train_fea[:, k]).reshape((len(test_fea), 1))),
                                              axis=1)


    train = np.concatenate([train_fea, train_label_rep.reshape((len(train_label_rep), 1))], axis=1)
    test = np.concatenate([test_fea, test_label_rep.reshape((len(test_label_rep), 1))], axis=1)

    train = np.concatenate([np.ones(num_train).reshape(num_train, 1), train], axis=1)
    test = np.concatenate([np.ones(num_test).reshape(num_test, 1), test], axis=1)
    return train, test, num_class


def non_linear_trans(orig_data, order):
    fea_num = orig_data.shape[1]
    if order >= 2:
        for i in range(fea_num):
            for j in np.arange(i, fea_num):
                orig_data = np.concatenate((orig_data, (orig_data[:, i] * orig_data[:, j]).reshape((len(orig_data), 1))), axis=1)

    elif order == 3:
        for i in range(fea_num):
            for j in np.arange(i, fea_num):
                for k in np.arange(j, fea_num):
                    orig_data = np.concatenate(
                        (orig_data, (orig_data[:, i] * orig_data[:, j] * orig_data[:, k]).reshape((len(orig_data), 1))), axis=1)
    return orig_data


def reflect_data(data):
    labels = data[:, -1]
    fea_ma = data[:, :-1]
    fea_ref = np.zeros(fea_ma.shape)
    for i in range(data.shape[0]):
        if labels[i] == 1:
            fea_ref[i] = fea_ma[i] * -1
        else:
            fea_ref[i] = fea_ma[i]
    ref_data = np.concatenate((fea_ref, labels.reshape(labels.shape[0], 1)), axis=1)
    return ref_data

class Binary_Perceptron(object):
    def __init__(self, args):
        self.lr = args.lr
        self.max_epoch = args.max_epoch

    def validate_train(self, data_fea):
        '''
        data_fea: shape [batch, fea_num+1]
        use reflected datapoints
        '''
        self.forward_cal = np.dot(self.weights, data_fea.T)
        acc = np.sum((self.forward_cal > 0)) / len(data_fea)
        return acc

    def validate(self, data_fea, labels):
        '''
        data_fea: shape [batch, fea_num+1]
        not use reflected datapoints
        '''
        self.forward_cal = np.dot(self.weights, data_fea.T)
        pred_label = (self.forward_cal <= 0)
        acc = np.sum((pred_label == labels)) / len(data_fea)
        return pred_label, acc

    def predictor(self, data_fea):
        forward_c = np.dot(self.weights, data_fea.T)
        pred_label = (forward_c <= 0)
        return pred_label

    def train(self, data_fea, labels, test_fea, test_labels, args):
        '''
        data_fea: augmented feature, shape [data_num, num_fea+1]
        self.weight shape [num_class, num_fea+1]
        train datas are reflected
        test datas are unreflected
        '''

        self.weights = np.ones(data_fea.shape[1])

        best_loss = math.inf
        best_weights = np.zeros(data_fea.shape[1])

        for epoch in range(self.max_epoch):
            if args.shuffle == True:
                idx = np.random.permutation(len(data_fea))
                data_fea = data_fea[idx]
                labels = labels[idx]
            for i in range(len(data_fea)):
                g_x = np.dot(self.weights, data_fea[i])
                if g_x <= 0:
                    self.weights += self.lr * data_fea[i]

                if epoch == self.max_epoch-1 and i >= len(data_fea)-100:
                    self.criterion(data_fea)
                    if self.loss <= best_loss:
                        best_loss = self.loss
                        best_weights = self.weights
            self.criterion(data_fea)
            train_acc = self.validate_train(data_fea)
            print(f'[Epoch {epoch + 1}][Train] \t Loss: {self.loss:.4e} \t Acc {train_acc:6.4f}')

            #self.criterion(test_fea, test_labels)
            _, test_acc = self.validate(test_fea, test_labels)
            print(f'[Epoch {epoch + 1}][Eval] \t Loss: --- \t Acc {test_acc:6.4f}')
        self.weights = best_weights

        self.criterion(data_fea)
        f_acc = self.validate_train(data_fea)
        print('Train End!')
        print(f'[Final][Train] \t Loss: {self.loss:.4e} \t Acc {f_acc:6.4f}')

        #self.criterion(test_fea, test_labels)
        _, f_test_acc = self.validate(test_fea, test_labels)
        print(f'[Final][Eval] \t Loss: --- \t Acc {f_test_acc:6.4f}')



    #augmented loss
    def criterion(self, data_fea):
        '''
        weights [num_class, num_fea+1]
        data_fea [data_num, num_fea+1]
        labels [data_num,]
        '''
        _ = self.validate_train(data_fea)
        self.loss = np.sum(self.forward_cal * (self.forward_cal <= 0)) * (-1)

    def get_weights(self, augment=True):
        if augment:
            return self.weights
        else:
            return self.weights[1:]



def train_top(args):
    #gen data
    train_data, test_data, num_class = gen_data(args)
    train_copy = train_data

    if args.reflect_data:
        train_data = reflect_data(train_data)

    train_fea = train_data[:, :-1]
    train_label = train_data[:, -1]
    test_fea = test_data[:, :-1]
    test_label = test_data[:, -1]

    # create model and train
    if args.repeat == 1:
        model = Binary_Perceptron(args)
        model.train(train_fea, train_label, test_fea, test_label, args)
        pred_test, _ = model.validate(test_fea, test_label)
        pred_train, _ = model.validate(train_copy[:, :-1], train_copy[:, -1])
        weights = model.get_weights(augment=True)

        order = args.nonlinear


        plotDecBoundaries_Nonlinear(train_copy[:, :-1][:, 1:args.fea_num+1], train_copy[:, -1].astype(np.int), non_linear_trans, model.predictor, order, fsize=(6, 4), legend_on=False)
        plotDecBoundaries_Nonlinear(test_fea[:, 1:args.fea_num+1], test_label.astype(np.int), non_linear_trans, model.predictor, order, fsize=(6, 4), legend_on=False)


    else:
        acc_train = []
        acc_test = []
        for re in range(args.repeat):
            print('Repeat ', re + 1, 'times!')
            model = Binary_Perceptron(args)
            model.train(train_fea, train_label, test_fea, test_label, args)
            ta_acc = model.validate_train(train_fea)
            _, tt_acc = model.validate(test_fea, test_label)
            acc_train.append(ta_acc)
            acc_test.append(tt_acc)

        print(f'Nonlinear Order: {args.nonlinear}')
        train_mean = np.mean(np.array(acc_train))
        train_std = np.sqrt(np.var(np.array(acc_train)))
        print('Train acc mean:', train_mean)
        print('Train acc std:', train_std)

        test_mean = np.mean(np.array(acc_test))
        test_std = np.sqrt(np.var(np.array(acc_test)))
        print('Test acc mean:', test_mean)
        print('Test acc std:', test_std)


    '''
    #save model
    outfile = f'./model/model_chenghao_li_binary_{args.binary_class}.hd5'
    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset('W', data=np.asarray(weights))
    '''



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multiclass perceptron')

    parser = init_parser(parser)

    args = parser.parse_args()

    train_top(args)
