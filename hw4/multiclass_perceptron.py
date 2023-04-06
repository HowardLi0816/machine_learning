import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import math
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    parser.add_argument('--binary_class', type=str2bool, default=True)
    parser.add_argument('--binary_num', type=int, default=2)
    parser.add_argument('--normalize', type=str2bool, default=True)
    parser.add_argument('--repeat', type=int, default=1)
    #parser.add_argument('--print_freq', type=int, default=50)

    return parser

def gen_data(args):
    '''
    output is augmented and normalized trian and test data with labels in the last column
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

class Multiclass_Perceptron(object):
    def __init__(self, num_class, args):
        self.lr = args.lr
        self.max_epoch = args.max_epoch
        self.num_class = num_class

    def validate(self, data_fea, labels):
        '''
        data_fea: shape [batch, fea_num+1]
        '''
        self.forward_cal = np.dot(self.weights, data_fea.T)
        pred_label = np.argmax(self.forward_cal, axis=0)
        acc = np.sum((pred_label == labels)) / len(data_fea)
        return pred_label, acc

    def train(self, data_fea, labels, test_fea, test_labels, args):
        '''
        data_fea: augmented feature, shape [data_num, num_fea+1]
        self.weight shape [num_class, num_fea+1]
        '''

        self.weights = np.ones((self.num_class, data_fea.shape[1]))

        best_loss = math.inf
        best_weights = np.zeros((self.num_class, data_fea.shape[1]))

        for epoch in range(self.max_epoch):
            if args.shuffle == True:
                idx = np.random.permutation(len(data_fea))
                data_fea = data_fea[idx]
                labels = labels[idx]
            for i in range(len(data_fea)):
                pred = np.argmax(np.dot(self.weights, data_fea[i]))
                if pred != labels[i]:
                    self.weights[int(labels[i])] += self.lr * data_fea[i]
                    self.weights[int(pred)] -= self.lr * data_fea[i]
                if epoch == self.max_epoch-1 and i >= len(data_fea)-100:
                    self.criterion(data_fea, labels)
                    if self.loss <= best_loss:
                        best_loss = self.loss
                        best_weights = self.weights
            self.criterion(data_fea, labels)
            _, train_acc = self.validate(data_fea, labels)
            print(f'[Epoch {epoch + 1}][Train] \t Loss: {self.loss:.4e} \t Acc {train_acc:6.4f}')

            self.criterion(test_fea, test_labels)
            _, test_acc = self.validate(test_fea, test_labels)
            print(f'[Epoch {epoch + 1}][Eval] \t Loss: {self.loss:.4e} \t Acc {test_acc:6.4f}')
        self.weights = best_weights

        self.criterion(data_fea, labels)
        _, f_acc = self.validate(data_fea, labels)
        print('Train End!')
        print(f'[Final][Train] \t Loss: {self.loss:.4e} \t Acc {f_acc:6.4f}')

        self.criterion(test_fea, test_labels)
        _, f_test_acc = self.validate(test_fea, test_labels)
        print(f'[Final][Eval] \t Loss: {self.loss:.4e} \t Acc {f_test_acc:6.4f}')



    #augmented loss
    def criterion(self, data_fea, labels):
        '''
        weights [num_class, num_fea+1]
        data_fea [data_num, num_fea+1]
        labels [data_num,]
        '''
        pred_label, _ = self.validate(data_fea, labels)
        self.loss = np.sum(self.forward_cal[pred_label.astype('int8'), np.arange(len(data_fea))] - self.forward_cal[labels.astype('int8'), np.argmax(len(data_fea))])

    def get_weights(self, augment=True):
        if augment:
            return self.weights
        else:
            return self.weights[:, 1:]


def train_top(args):
    #gen data
    train_data, test_data, num_class = gen_data(args)
    train_fea = train_data[:, :-1]
    train_label = train_data[:, -1]
    test_fea = test_data[:, :-1]
    test_label = test_data[:, -1]

    if args.repeat == 1:
        # create model and train
        model = Multiclass_Perceptron(num_class, args)
        model.train(train_fea, train_label, test_fea, test_label, args)
        pred_test, _ = model.validate(test_fea, test_label)

        # get weight vector
        weights = model.get_weights(augment=True)
        print('final weight vector:', weights)
        norm_weights = np.linalg.norm(weights, axis=1)
        print('final weight magnitude:', norm_weights)

        # confusion matrix on test set
        con_matrix = confusion_matrix(test_label, pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=con_matrix)
        disp.plot()
        plt.show()

    else:
        acc_list = []
        weight_norm_list = []
        con_matrix_list = []
        for re in range(args.repeat):
            print('Repeat ', re+1, 'times!')
            # create model and train
            model = Multiclass_Perceptron(num_class, args)
            model.train(train_fea, train_label, test_fea, test_label, args)
            pred_test, test_acc = model.validate(test_fea, test_label)
            acc_list.append(test_acc)

            # get weight vector
            weights = model.get_weights(augment=True)
            norm_weights = np.linalg.norm(weights, axis=1)
            weight_norm_list.append(norm_weights)

            # confusion matrix on test set
            con_matrix = confusion_matrix(test_label, pred_test)
            con_matrix_list.append(con_matrix)

        acc_mean = np.mean(np.array(acc_list))
        acc_std = np.sqrt(np.var(np.array(acc_list)))
        print('acc mean:', acc_mean)
        print('acc std:', acc_std)

        w_mean = np.mean(np.array(weight_norm_list), axis=0)
        w_std = np.sqrt(np.var(np.array(weight_norm_list), axis=0))
        print('weight norm mean:', w_mean)
        print('weight norm std:', w_std)

        con_mean = np.mean(np.array(con_matrix_list), axis=0)
        con_std = np.sqrt(np.var(np.array(con_matrix_list), axis=0))
        print('confusion matrix mean:', con_mean)
        print('confusion matrix std:', con_std)





if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Multiclass perceptron')

    parser = init_parser(parser)

    args = parser.parse_args()

    train_top(args)
