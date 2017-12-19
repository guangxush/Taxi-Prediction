# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing

import numpy as np
import pandas as pd
import os


def save_epoch(nn_model, epoch):
    if not os.path.exists('mlpmodels/'):
        os.makedirs('mlpmodels/')
    nn_model.save_weights('mlpmodels/weights_epoch_%d.h5' % epoch, overwrite=True)


def load_epoch(nn_model, epoch):
    assert os.path.exists('mlpmodels/weights_epoch_%d.h5' % epoch), 'Weights at epoch %d not found' % epoch
    nn_model.load_weights('mlpmodels/weights_epoch_%d.h5' % epoch)

	
seed = 7
np.random.seed(seed)
h = FeatureHasher(n_features=2048)
vec = DictVectorizer()
le = preprocessing.LabelEncoder()
nb_epoch = 100
batch_size = 2048

attr_name = ['taxiID', 'point', 'direction', 'time', 'duration', 'gridID']
train = pd.read_csv("train.txt", header=None)
train_set = train.values[:, [0, 1, 2, 3, 4, 5, 6]]
print(train_set[0])

test = pd.read_csv("test.txt")
test_set = test.values[:, [0, 1, 2, 3, 4, 5, 6]]
print(test_set[0])

dataset = train.values[:, [0, 1, 2, 3, 4, 5]]

samples = list()

for sample in dataset:
    sample_dict = dict()
    for index, attr in enumerate(sample):
        sample_dict[attr_name[index]] = attr
    samples.append(sample_dict)

h.fit(samples)

X_train = list()
y_train = list()

X_test = list()
y_test = list()
for sample in train_set:
    sample_dict = dict()
    for index, attr in enumerate(sample):
        attr = str(attr)
        if index == 6:
            y_train.append(int(attr))
            continue
        sample_dict[attr_name[index]] = attr
    X_train.append(sample_dict)

for sample in test_set:
    sample_dict = dict()
    for index, attr in enumerate(sample):
        attr = str(attr)
        if index == 6:
            y_test.append(int(attr))
            continue
        sample_dict[attr_name[index]] = attr
    X_test.append(sample_dict)

X_train = h.transform(X_train).toarray()
X_test = h.transform(X_test).toarray()
print(X_train[0])
print(X_test[0])
print(X_train.shape)
print(X_test.shape)

y_train = np.asarray(y_train, dtype='int16')
y_test = np.asarray(y_test, dtype='int16')
nb_classes = np.max(y_train) + 1
nb_test_classes = np.max(y_test) + 1
print('nb_classes: ', nb_classes)
print('nb_test_classes: ', nb_test_classes)

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test, nb_classes)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Dense(2048, input_dim=X_train.shape[1], init='uniform', activation='relu'))
model.add(Dense(1024, init='uniform', activation='relu'))
model.add(Dense(nb_classes, init='uniform', activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

hist = model.fit(X_train, y_train, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1, validation_data=(X_test, y_test))

#scores = model.evaluate(X_test,y_test,verbose=0)
#print("Model Accuracy: %.2%" % (scores[1]*100))

fl = open('mlp_loss.txt','w')
for loss in hist.history['loss']:
	fl.write(repr(loss) + '\n')

fa = open('mlp_acc.txt','w')
for acc in hist.history['acc']:
	fa.write(repr(acc) + '\n')

el = open('evmlp_loss.txt','w')
for loss in hist.history['val_loss']:
	el.write(repr(loss)+'\n')
ea = open('evmlp_acc.txt','w')
for acc in hist.history['val_acc']:
	ea.write(repr(acc)+'\n')

# # save_epoch(model, nb_epoch)
#
# # load_epoch(model, nb_epoch)
# # results = model.predict_classes(X_test, batch_size=128)
# # for index, product in enumerate(list(le.inverse_transform(results))):
# #     print(index, product)
# # print(results[0:2])
