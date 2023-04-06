import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import h5py

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
                        choices=['dry_bean'],
                        default='dry_bean')
    parser.add_argument('--max_epoch', default=100, type=int)
    parser.add_argument('--lr', default=1, type=float)
    #parser.add_argument('--min_lr', default=1e-3, type=float)
    #parser.add_argument('--warmup', default=10, type=int, help='number of warmup epochs')
    #parser.add_argument('--adjust_lr', default=False, type=str2bool)
    parser.add_argument('--shuffle', default=True, type=str2bool)
    parser.add_argument('--train_method', type=str, choices=['SGD'], default='SGD')
    parser.add_argument('--binary_class', type=int, default=2)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--reflect_data', type=str2bool, default=True)
    parser.add_argument('--plot_hist', type=str2bool, default=True)
    parser.add_argument('--mode', type=str.lower, choices=['binary_train', 'multi_eval'], default='binary_train')
    parser.add_argument('--decision_rule', type=str.lower, choices=['intersection', 'mvm1', 'mvm2'], default='intersection')
    #parser.add_argument('--print_freq', type=int, default=50)

    return parser

def gen_data(args):
    '''
    output is augmented and normalized trian and test data with labels in the last column, unreflected data
    '''
    if args.dataset == 'dry_bean':
        train_path = './data/Dry_Bean_train.csv'
        test_path = './data/Dry_Bean_test.csv'

        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_fea = train_data.drop("Class", axis=1)
        test_fea = test_data.drop("Class", axis=1)

        train_label = train_data["Class"]
        test_label = test_data["Class"]

        num_train = len(train_fea)
        num_test = len(test_fea)

        le = LabelEncoder()
        le.fit(train_label)

        num_class = len(le.classes_)
        train_label_rep = le.transform(train_label)
        test_label_rep = le.transform(test_label)

        if args.mode == 'binary_train':
            train_label_rep = (train_label_rep != args.binary_class)
            test_label_rep = (test_label_rep != args.binary_class)


        #normalize the data
        if args.normalize:
            scalar = StandardScaler()
            scalar.fit(train_fea)
            train_fea = scalar.transform(train_fea)
            test_fea = scalar.transform(test_fea)

        train = np.concatenate([train_fea, train_label_rep.reshape((len(train_label_rep), 1))], axis=1)
        test = np.concatenate([test_fea, test_label_rep.reshape((len(test_label_rep), 1))], axis=1)

        train = np.concatenate([np.ones(num_train).reshape(num_train, 1), train], axis=1)
        test = np.concatenate([np.ones(num_test).reshape(num_test, 1), test], axis=1)
    return train, test, num_class

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

    def reload_weights(self, weights):
        self.weights = weights


def train_top(args):
    #gen data
    train_data, test_data, num_class = gen_data(args)

    if args.reflect_data:
        train_data = reflect_data(train_data)

    train_fea = train_data[:, :-1]
    train_label = train_data[:, -1]
    test_fea = test_data[:, :-1]
    test_label = test_data[:, -1]

    # create model and train
    model = Binary_Perceptron(args)
    model.train(train_fea, train_label, test_fea, test_label, args)
    pred_test, _ = model.validate(test_fea, test_label)
    weights = model.get_weights(augment=True)

    #save model
    outfile = f'./model/model_chenghao_li_binary_{args.binary_class}.hd5'
    with h5py.File(outfile, 'w') as hf:
        hf.create_dataset('W', data=np.asarray(weights))

    if args.plot_hist:
        g_x = np.dot(weights, test_fea.T)
        g_x_0 = g_x[g_x>0]
        g_x_1 = g_x[g_x<=0]
        plt.hist(g_x_0, bins=200, label='class'+str(args.binary_class), color='r', alpha=0.5)
        plt.hist(g_x_1, bins=200, label='other class', color='g', alpha=0.5)
        plt.legend()
        plt.show()

def multi_eval(args):
    # gen data
    train_data, test_data, num_class = gen_data(args)
    train_fea = train_data[:, :-1]
    train_label = train_data[:, -1]
    test_fea = test_data[:, :-1]
    test_label = test_data[:, -1]

    #load model weight
    mul_weights = []
    for cla in range(num_class):
        file_name = f'./model/model_chenghao_li_binary_{cla}.hd5'
        with h5py.File(file_name, 'r') as hf:
            weights = hf['W'][:]
        mul_weights.append(weights)
    mul_weights = np.array(mul_weights)

    train_pred = np.dot(mul_weights, train_fea.T)
    test_pred = np.dot(mul_weights, test_fea.T)

    if args.decision_rule == 'intersection':
        tmp = np.sum((train_pred > 0), axis=0)
        determined_point = (tmp == 1)
        pred_result = -1 * np.ones(train_pred.shape[1])
        pred_result = pred_result + determined_point * (np.argmax(train_pred, axis=0) + 1)

        tmp_tt = np.sum((test_pred > 0), axis=0)
        determined_point_tt = (tmp_tt == 1)
        pred_result_tt = -1 * np.ones(test_pred.shape[1])
        pred_result_tt = pred_result_tt + determined_point_tt * (np.argmax(test_pred, axis=0) + 1)

    elif args.decision_rule == 'mvm1':
        pred_result = np.argmax(train_pred, axis=0)

        pred_result_tt = np.argmax(test_pred, axis=0)


    elif args.decision_rule == 'mvm2':
        norm_weights = np.tile(np.linalg.norm(mul_weights[:, 1:], axis=1).reshape((num_class,1)), train_pred.shape[1])
        pred_result = np.argmax(train_pred/norm_weights, axis=0)

        norm_weights_tt = np.tile(np.linalg.norm(mul_weights[:, 1:], axis=1).reshape((num_class, 1)), test_pred.shape[1])
        pred_result_tt = np.argmax(test_pred/norm_weights_tt, axis=0)



    acc, err, uncla = validate_with_unclassify(pred_result, train_label)
    acc_tt, err_tt, uncla_tt = validate_with_unclassify(pred_result_tt, test_label)

    print(f'decision_rule: {args.decision_rule}')
    print(f'[train] \t Acc: {acc:6.4f} \t Err: {err:6.4f} \t Unclassify: {uncla:6.4f}')
    print(f'[test] \t Acc: {acc_tt:6.4f} \t Err: {err_tt:6.4f} \t Unclassify: {uncla_tt:6.4f}')


def validate_with_unclassify(pred_results, labels):
    unclassify = np.sum((pred_results<0)) / len(pred_results)
    acc = np.sum((pred_results == labels)) / len(pred_results)
    err = 1-acc-unclassify
    return acc, err, unclassify

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multiclass perceptron')

    parser = init_parser(parser)

    args = parser.parse_args()

    if args.mode == 'binary_train':
        train_top(args)
    elif args.mode == 'multi_eval':
        multi_eval(args)

