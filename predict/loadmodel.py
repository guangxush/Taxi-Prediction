#coding:utf-8
import h5py
import numpy as np
import os
from keras.models import load_model
import sys
import pandas as pd
from collections import Counter
from keras.layers import Input, Embedding, Dense, Reshape, Dropout
from keras.models import Model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

def load_epoch(nn_model, epoch):
    assert os.path.exists('../submissions/mlpmodels/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
    nn_model.load_weights('../submissions/mlpmodels/weights_epoch_%d.h5' % epoch)


def loadmodel_embedding(file_path):
    np.random.seed(7)
    batch_size = 2048
    print (os.path.join(file_path, 'test-weigts.02-0.29.hdf5'))
    model = load_model('../submissions/embeddingmodels/test-weigts.02-0.29.hdf5')
    # test input
    test_tid, test_grid, test_direction, test_tStamp, test_dur, test_dis = [], [], [], [], [], []
    test_target = []
    test_data = open('../data/test.txt', 'r')
    # generate test
    nb_classes = 2823
    for line in test_data:
        tmp = line.strip().split(',')
        test_tid.append(tmp[0])
        test_grid.append(tmp[1])
        test_direction.append(tmp[2])
        test_tStamp.append(tmp[3])
        test_dur.append(tmp[4])
        test_dis.append(tmp[5])
        test_target.append(tmp[6])

    # add index
    fulList = test_tid + test_grid + test_direction + test_tStamp + test_dur + test_dis
    ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))
    # to index
    to_idx = lambda x: [ful2idx[word] for word in x]
    test_tid_array = to_idx(test_tid)
    test_grid_array = to_idx(test_grid)
    test_direction_array = to_idx(test_direction)
    test_tStamp_array = to_idx(test_tStamp)
    test_dur_array = to_idx(test_dur)
    test_dis_array = to_idx(test_dis)

    test_in_array = np.column_stack(
        (test_tid_array, test_grid_array, test_direction_array, test_tStamp_array, test_dur_array, test_dis_array))
    test_target_array = np.asarray(test_target, dtype='int16')
    test_target_array = np_utils.to_categorical(test_target_array, nb_classes)
    prob = model.predict(test_in_array, batch_size=batch_size)
    print('test after load: ', prob)
    scores = model.evaluate(test_in_array, test_target_array, verbose=0)
    print("Model Accuracy:%.2f%%" % (scores[1] * 100))
    print("Model Loss:%.2f%%" % (scores[0] * 100))

def loadmodel_mlp(file_path):
    #model = load_model('../submissions/mlpmodels/weights_epoch_100.h5')
    seed = 7
    np.random.seed(seed)
    h = FeatureHasher(n_features=2048)
    vec = DictVectorizer()
    le = preprocessing.LabelEncoder()
    nb_epoch = 100
    batch_size = 2048
    nb_classes = 2823

    attr_name = ['taxiID', 'point', 'direction', 'time', 'duration', 'gridID']

    test = pd.read_csv("../data/test.txt")
    test_set = test.values[:, [0, 1, 2, 3, 4, 5, 6]]
    print(test_set[0])

    X_test = list()
    y_test = list()
    for sample in test_set:
        sample_dict = dict()
        for index, attr in enumerate(sample):
            attr = str(attr)
            if index == 6:
                y_test.append(int(attr))
                continue
            sample_dict[attr_name[index]] = attr
        X_test.append(sample_dict)

    X_test = h.transform(X_test).toarray()
    print(X_test[0])
    print(X_test.shape)

    y_test = np.asarray(y_test, dtype='int16')
    nb_test_classes = np.max(y_test) + 1
    print('nb_test_classes: ', nb_test_classes)

    y_test = np_utils.to_categorical(y_test, nb_classes)
    print(y_test.shape)

    model = Sequential()
    model.add(Dense(2048, input_dim=X_test.shape[1], init='uniform', activation='relu'))
    model.add(Dense(1024, init='uniform', activation='relu'))
    model.add(Dense(nb_classes, init='uniform', activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    #hist = model.fit(X_test, y_test, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
                     #validation_data=(X_test, y_test))

    load_epoch(model, nb_epoch)
    results = model.predict_classes(X_test, batch_size=128)

    print(results[0:2])

if __name__ == '__main__':
    if sys.argv[1] == 'embedding':
        loadmodel_embedding(file_path='../submissions/embeddingmodels')
    elif sys.argv[1] == 'mlp':
        loadmodel_mlp(file_path='../submissions/mlpmodels')