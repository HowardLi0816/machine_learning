import numpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from imblearn.over_sampling import SMOTENC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def gen_data(PTH):
    '''
    get data, not use reflected data points nor perform any standardization or any data preprocessing, no augmented
    '''
    train_path = f'{PTH}/credit_card_dataset_train.csv'
    test_path = f'{PTH}/credit_card_dataset_test.csv'

    train_data = pd.read_csv(train_path).values
    test_data = pd.read_csv(test_path).values

    train_data = data_clean(train_data)
    test_data = data_clean(test_data)

    return train_data, test_data


def data_clean(dataset):
    # for x3: Education
    x3 = dataset[:, 2]
    res = (x3 != 1) * (x3 != 2) * (x3 != 3)
    dataset[res, 2] = 4
    # for x4: Marital status
    x4 = dataset[:, 3]
    res = (x3 != 1) * (x3 != 2)
    dataset[res, 3] = 3
    # for x6-x11: History of past payment
    dataset[:, 5:11] = dataset[:, 5:11] + 1
    dataset = dataset.astype(numpy.float64)
    return dataset


def normalization(train_data, test_data):
    train = train_data[:, :-1]
    test = test_data[:, :-1]
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)
    train_data[:, :-1] = (train - mean) / std
    test_data[:, :-1] = (test - mean) / std

    return train_data, test_data


def data_augment(dataset, no_change=range(1, 11)):
    smote = SMOTENC(categorical_features=no_change)
    X_resampled, y_resampled = smote.fit_resample(dataset[:, :-1], dataset[:, -1])
    res = np.zeros([X_resampled.shape[0], X_resampled.shape[1] + 1])
    res[:, :-1] = X_resampled
    res[:, -1] = y_resampled
    return res


def poly_transform(dataset, degree=1):
    poly = PolynomialFeatures(degree=degree)
    poly_dataset = poly.fit_transform(dataset[:, :-1])
    result = np.concatenate((poly_dataset, dataset[:, -1].reshape((len(dataset), 1))), axis=1)
    return result

def PCA_transform(trainset, testset, n_components):
    pca = PCA(n_components=n_components)
    pca_dataset = pca.fit_transform(trainset[:, :-1])
    train_result = np.concatenate((pca_dataset, trainset[:, -1].reshape((len(trainset), 1))), axis=1)
    test_fea = pca.transform(testset[:, :-1])
    test_result = np.concatenate((test_fea, testset[:, -1].reshape((len(testset), 1))), axis=1)

    return train_result, test_result

def LDA_transform(trainset, testset, n_components=None):
    lda = LinearDiscriminantAnalysis(n_components=n_components)
    lda_dataset = lda.fit_transform(trainset[:, :-1], trainset[:, -1])
    train_result = np.concatenate((lda_dataset, trainset[:, -1].reshape((len(trainset), 1))), axis=1)
    test_fea = lda.transform(testset[:, :-1])
    test_result = np.concatenate((test_fea, testset[:, -1].reshape((len(testset), 1))), axis=1)

    return train_result, test_result


def UFS_reduce(trainset, testset, k):
    ufs = SelectKBest(f_classif, k=k)
    ufs_dataset = ufs.fit_transform(trainset[:, :-1], trainset[:, -1])
    train_result = np.concatenate((ufs_dataset, trainset[:, -1].reshape((len(trainset), 1))), axis=1)
    test_fea = ufs.transform(testset[:, :-1])
    test_result = np.concatenate((test_fea, testset[:, -1].reshape((len(testset), 1))), axis=1)

    return train_result, test_result

def RFE_reduce(trainset, testset, n_features_to_select):
    # estimator = SVR(kernel="linear")
    lr = LinearRegression()
    selector = RFE(lr, n_features_to_select=n_features_to_select, step=1)
    rfe_dataset = selector.fit_transform(trainset[:, :-1], trainset[:, -1])
    train_result = np.concatenate((rfe_dataset, trainset[:, -1].reshape((len(trainset), 1))), axis=1)
    test_fea = selector.transform(testset[:, :-1])
    test_result = np.concatenate((test_fea, testset[:, -1].reshape((len(testset), 1))), axis=1)

    return train_result, test_result

def SFS_reduce(trainset, testset, k):
    # knn = KNeighborsClassifier(n_neighbors=3)
    lr = LinearRegression()
    sfs = SequentialFeatureSelector(lr, n_features_to_select=k)
    sfs.fit(trainset[:, :-1], trainset[:, -1])
    X_train_selected = sfs.transform(trainset[:, :-1])
    train_result = np.concatenate((X_train_selected, trainset[:, -1].reshape((len(trainset), 1))), axis=1)
    X_test_selected = sfs.transform(testset[:, :-1])
    test_result = np.concatenate((X_test_selected, testset[:, -1].reshape((len(testset), 1))), axis=1)

    return train_result, test_result


def gen_data_2(PTH, args):
    train_data, test_data = gen_data(PTH)

    if args.norm:
        train_data, test_data = normalization(train_data, test_data)
    if args.data_augment:
        train_data = data_augment(train_data)
    if args.ufs:
        train_data, test_data = UFS_reduce(train_data, test_data, k=args.ufs_k)
    if args.rfe:
        train_data, test_data = RFE_reduce(train_data, test_data, n_features_to_select=args.rfe_k)
    if args.sfs:
        train_data, test_data = SFS_reduce(train_data, test_data, k=args.sfs_k)
    if args.pca:
        train_data, test_data = PCA_transform(train_data, test_data, n_components=args.pca_n_component)
    if args.lda:
        train_data, test_data = LDA_transform(train_data, test_data, n_components=args.lda_n_component)
    if args.non_linear:
        train_data = poly_transform(train_data, degree=args.poly)
        test_data = poly_transform(test_data, degree=args.poly)

    return train_data, test_data


def train_val_split(dataset):
    X_train, X_val, y_train, y_val = train_test_split(dataset[:, :-1], dataset[:, -1], test_size=0.1)
    train_result = np.concatenate((X_train, y_train.reshape((len(y_train), 1))), axis=1)
    val_result = np.concatenate((X_val, y_val.reshape((len(y_val), 1))), axis=1)
    return train_result, val_result





