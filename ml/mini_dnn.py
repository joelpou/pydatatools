from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from matplotlib import pyplot
from numpy import where
import numpy as np
import pandas as pd
import sys

devCsv = str(sys.argv[1])
trainCsv = str(sys.argv[2])
testCsv = str(sys.argv[3])


def prepare_data(name, csv_path, scaled=True):
    labels = []
    print('')
    print(csv_path)
    df = pd.read_csv(csv_path)

    df = shuffle(df)
    df.to_csv(name + '.csv')

    for i in range(df.shape[0]):
        if df.iloc[i, 0] == 1:
            labels.append(1)
        else:
            labels.append(-1)

    y = np.array(labels).astype(int)

    # df.to_csv('label' + name + '.csv')

    df = df.drop(df.columns[[0, 1]], axis=1).astype('float64')
    X = df.to_numpy()

    if scaled:
        X = StandardScaler().fit_transform(X)
        dfn = pd.DataFrame.from_records(X)
        dfn.to_csv('out_features.csv')
    return X, y


devX, devy = prepare_data('dev', devCsv, False)
print('')
# print(devX)
print(devX.shape)
# print(devy)
print(devy.shape)

trainX, trainy = prepare_data('train', trainCsv, False)
print('')
# print(trainX)
print(trainX.shape)
# print(trainy)
print(trainy.shape)

testX, testy = prepare_data('test', testCsv, False)
print('')
# print(testX)
print(testX.shape)
# print(testy)
print(testy.shape)

# # generate 2d classification dataset
# X, y = make_circles(n_samples=1000, noise=0.1, random_state=1)
# print(X.shape)
# print(type(X))
# print(y.shape)
# print(type(y))
# # change y from {0,1} to {-1,1}
# y[where(y == 0)] = -1
#
# # split into train and test
# n_train = 500
# trainX, testX = X[:n_train, :], X[n_train:, :]
# trainy, testy = y[:n_train], y[n_train:]

# Classification
model = Sequential()
model.add(Dense(9, input_dim=18, activation='relu', kernel_initializer='he_uniform'))  # 36, 18, relu, he_uniform
# model.add(Dense(1, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
#opt = SGD(lr=0.01, momentum=0.9, decay=0.01)  # optimization, same
opt = SGD(lr=0.001, momentum=0.9, decay=0.01)  # optimization, same
# opt = Adam(lr=0.01, clipvalue=0.5)
# model.compile(loss='hinge', optimizer=opt, metrics=['accuracy'])
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
# fit model
history = model.fit(trainX, trainy, validation_data=(devX, devy), epochs=10000, verbose=1)  # train/val data, epochs=10,000, verbose = 1, set dev
# evaluate the model
# _, train_acc = model.evaluate(trainX, trainy, verbose=1)    #stochastic gradient descent, ver como se comporta data, 85% de accuracy
_, test_acc = model.evaluate(testX, testy, verbose=1)
# print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))    # more epochs, more accuracy
print('Test: %.3f' % (test_acc))  # more epochs, more accuracy
# plot loss during training
pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
# plot accuracy during training
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend()
pyplot.show()

# future work:
# cross-validation


# df_ = pd.DataFrame()
# df_ = df_.fillna(0)

# for file in glob.glob(devDir + '**/*.csv', recursive=True):
#     if os.path.isfile(file):
#         print('\n' + file + '\n')
#         df0 = pd.read_csv(file)
#         dfL.append(df0)
# print(df.shape[0])    #get label

# if df.iloc[0, 0] == 1:
#     # testy.append(np.ones((df.shape[0], 1), dtype=int))
#     for i in range(df.shape[0]):
#         testy.append(1)
# else:
#     for i in range(df.shape[0]):
#         testy.append(-1)
# print(dfL)
