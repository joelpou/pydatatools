import keras
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from numpy import where
import numpy as np
import pandas as pd
import sys

testCsv = str(sys.argv[1])


def prepare_data(name, csv_path, shuf, scal):
    labels = []
    print('')
    print(csv_path)
    df = pd.read_csv(csv_path)

    if shuf:
        df = shuffle(df)

    # df.to_csv(name + '.csv')

    for i in range(df.shape[0]):
        if df.iloc[i, 0] == 1:
            labels.append(1)
        else:
            labels.append(0)

    y = np.array(labels).astype(int)

    # drop label and name column
    df = df.drop(df.columns[[0, 1]], axis=1).astype('float32')
    X = df.to_numpy()

    if scal:
        X = StandardScaler().fit_transform(X)
        dfn = pd.DataFrame.from_records(X)
        # dfn.to_csv('out_features_scaled.csv')

    print('')
    # print(X)
    print('X_' + name + '_dim: ' + str(X.shape))
    # print(y)
    print('y_' + name + '_dim: ' + str(y.shape))

    return X, y


def get_classifier():
    classfr = Sequential()
    classfr.add(Dense(9, input_dim=18, activation='relu'))
    classfr.add(Dense(3, activation='relu'))
    classfr.add(Dense(9, activation='relu'))
    classfr.add(Dense(2, activation='softmax'))  # real: [1, 0], spoof: [0, 1]
    return classfr


def to_categorical_fixed(testy):
    testy_one_hot = np.zeros((testy.size, 2))
    for i in range(testy.size):
        if testy[i] == 1:
            testy_one_hot[i][0] = 1
        else:
            testy_one_hot[i][1] = 1
    return testy_one_hot


testX, testyy = prepare_data('test', testCsv, False, False)

testy_one_hotx = to_categorical_fixed(testyy)

classifier = get_classifier()
classifier.load_weights('classifier_weights.h5')
predicted_y = classifier.predict(testX)
predicted_y = np.rint(predicted_y)

res = np.sum(np.absolute(testy_one_hotx - predicted_y.astype(float)), axis=1)
items_correct = res[res == 0]
test_acc = float(items_correct.size) / float(res.size)
print('Test: %.3f' % test_acc)
predicted_y = predicted_y.astype(float)
realsamples_expected = np.where(testy_one_hotx[:, 0] == 1)
realsamples_actual = predicted_y[realsamples_expected[0]]
realsamples_spoofs = np.where(realsamples_actual[:, 0] == 0)
bpcer = float(len(realsamples_spoofs[0])) / float(len(realsamples_expected[0]))  # real misclassified as spoof

spoofsamples_expected = np.where(testy_one_hotx[:, 1] == 1)
spoofsamples_actual = predicted_y[spoofsamples_expected[0]]
spoofsamples_real = np.where(spoofsamples_actual[:, 1] == 0)
apcer = float(len(spoofsamples_real[0])) / float(len(spoofsamples_expected[0]))  # spoof misclassified as real

print('bpcer: {:.3f}, realsamples expected: {:d}, realsamples mistaken for spoofs: {:d}'.format(bpcer, len(
    realsamples_expected[0]), len(realsamples_spoofs[0])))
print('apcer: {:.3f}, spoofsamples expected: {:d}, spoofsamples mistaken for real: {:d}'.format(apcer, len(
    spoofsamples_expected[0]), len(spoofsamples_real[0])))
