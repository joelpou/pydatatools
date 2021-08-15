import keras
import tensorflow as tf
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Layer
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


class DenseTranspose(Layer):
    def __init__(self, dense, activation=None, **kwargs):
        self.biases = self.add_weight(name="bias", initializer="zeros",
                                      shape=[self.dense.input_shape[-1]])
        self.dense = dense
        self.activation = keras.activations.get(activation)
        super().__init__(**kwargs)

    def build(self, batch_input_shape):
        super().build(batch_input_shape)

    def call(self, inputs, **kwargs):
        z = tf.matmul(inputs, self.dense.weights[0], transpose_b=True)
        return self.activation(z + self.biases)


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
    df = df.drop(df.columns[[0, 1]], axis=1).astype('float64')
    X = df.to_numpy()

    if scal:
        X = StandardScaler().fit_transform(X)
        dfn = pd.DataFrame.from_records(X)
        dfn.to_csv('out_features_scaled.csv')

    print('')
    # print(X)
    print('X_' + name + '_dim: ' + str(X.shape))
    # print(y)
    print('y_' + name + '_dim: ' + str(y.shape))

    return X, y


def to_categorical_fixed(testy):
    testy_one_hot = np.zeros((testy.size, 2))
    for i in range(testy.size):
        if testy[i] == 1:
            testy_one_hot[i][0] = 1
        else:
            testy_one_hot[i][1] = 1
    return testy_one_hot


def get_encoder():
    encoder = Sequential()
    encoder.add(Dense(9, input_dim=18, activation='selu'))
    encoder.add(Dense(3, activation='selu'))
    return encoder


def get_decoder():
    decoder = Sequential()
    decoder.add(Dense(9, input_dim=3, activation='selu'))
    decoder.add(Dense(18, activation='linear'))
    return decoder


def get_classifier():
    classfr = Sequential()
    classfr.add(Dense(9, input_dim=18, activation='selu'))
    classfr.add(Dense(3, activation='selu'))
    classfr.add(Dense(9, activation='selu'))
    classfr.add(Dense(2, activation='softmax'))  # real: [1, 0], spoof: [0, 1]
    return classfr


def generate_autoencoder(encoder, train_x, dev_x, test_x):
    decoder = get_decoder()

    model = Sequential([encoder, decoder])

    opt = SGD(lr=0.1)

    model.compile(loss='mean_squared_error', optimizer=opt)
    model.fit(train_x,
              train_x,
              epochs=30,
              # batch_size=256,  # 256 first try
              shuffle=False,
              validation_data=[dev_x, dev_x],
              verbose=1)

    # evaluate the model
    encoded_iqm = encoder.predict(test_x)
    decoded_iqm = decoder.predict(encoded_iqm)

    # average_error = np.sum((decoded_iqm - testX)*(decoded_iqm - testX)) / len(decoded_iqm)
    average_error = (np.square(decoded_iqm - test_x)).mean(axis=1)

    print('average_error: ' + str(average_error))

    # stacked_encoder.save('autoencoder_encoder.h5')
    encoder.save_weights('autoencoder_weights.h5')


devX, devy = prepare_data('dev', devCsv, True, True)
trainX, trainy = prepare_data('train', trainCsv, True, True)
testX, testyy = prepare_data('test', testCsv, False, False)

# AutoEncoder: 18->9->3->9->18, decoding part is to train (learn pattern), for classification only use encoder part
# 18->9->3->1 encoding part + if real or spoof classifier

stacked_encoder = get_encoder()

if True:
    generate_autoencoder(stacked_encoder, trainX, devX, testX)

stacked_encoder.load_weights('autoencoder_weights.h5')

classifier = get_classifier()

for l1, l2 in zip(classifier.layers[0:3], stacked_encoder.layers[:]):
    l1.set_weights(l2.get_weights())

for layer in classifier.layers[0:3]:
    layer.trainable = False

classifier.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

trainy_one_hot = to_categorical_fixed(trainy)
devy_one_hot = to_categorical_fixed(devy)

print('Original label:', trainy[0])
print('After conversion to one-hot:', trainy_one_hot[0])

history = classifier.fit(trainX,
                         trainy_one_hot,
                         # batch_size=256,
                         validation_data=(devX, devy_one_hot),
                         epochs=30,
                         shuffle=False,
                         verbose=1)

classifier.save_weights('classifier_weights.h5')

testy_one_hotx = to_categorical_fixed(testyy)

# _, test_acc = classifier.evaluate(testX, testy_one_hot, verbose=1)
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
