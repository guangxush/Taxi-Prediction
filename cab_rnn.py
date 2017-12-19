# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import SimpleRNN as RNN
from keras.utils import np_utils

import numpy as np
import os

# fix random seed for reproducibility
np.random.seed(7)
# define the raw dataset

if not os.path.exists('rnnmodels'):
    os.makedirs('rnnmodels')

nb_epoch = 100
batch_size = 2048
status_maxlen = 4
nb_classes = 1823

# input factors
tid, grid, direction, tStamp = [], [], [], []

# test input
test_tid, test_grid, test_direction, test_tStamp = [], [], [], []

# result classification
target = []
test_target = []

# read file
fr = open('400_7Days.txt', 'rb')
test_data = open('400_test.txt', 'rb')

# generate train
for line in fr:
    tmp = line.strip().split(',')
    tid.append((tmp[0]))
    grid.append(tmp[1])
    direction.append(tmp[2])
    tStamp.append(tmp[3])

    target.append(tmp[4])

# add index
fulList = tid + grid + direction + tStamp

ful2idx = dict((v, i) for i, v in enumerate(list(set(fulList))))

n_status = len(ful2idx.keys())

# to index
to_idx = lambda x: [ful2idx[word] for word in x]

tid_array = to_idx(tid)
grid_array = to_idx(grid)
direction_array = to_idx(direction)
tStamp_array = to_idx(tStamp)

in_array = np.column_stack((tid_array, grid_array, direction_array, tStamp_array))
in_array = np.reshape(in_array, (in_array.shape[0], status_maxlen, 1))
in_array = in_array / float(n_status)
target_array = np.asarray(target, dtype='int16')
target_array = np_utils.to_categorical(target_array, nb_classes)

# generate test
for line in test_data:
    tmp = line.strip().split(',')
    test_tid.append(tmp[0])
    test_grid.append(tmp[1])
    test_direction.append(tmp[2])
    test_tStamp.append(tmp[3])

    test_target.append(tmp[4])

test_tid_array = to_idx(test_tid)
test_grid_array = to_idx(test_grid)
test_direction_array = to_idx(test_direction)
test_tStamp_array = to_idx(test_tStamp)

test_in_array = np.column_stack((test_tid_array, test_grid_array, test_direction_array, test_tStamp_array))
test_in_array = np.reshape(test_in_array, (test_in_array.shape[0], status_maxlen, 1))
test_in_array = test_in_array / float(n_status)
test_target_array = np.asarray(test_target, dtype='int16')
test_target_array = np_utils.to_categorical(test_target_array, nb_classes)

# create and fit the model
model = Sequential()
model.add(RNN(512, input_shape=(in_array.shape[1], 1)))
model.add(Dense(target_array.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(in_array, target_array, nb_epoch=nb_epoch, batch_size=batch_size, verbose=1,
          validation_data=(test_in_array, test_target_array))
# summarize performance of the model
scores = model.evaluate(test_in_array, test_target_array, verbose=0)
print("Model Accuracy: %.2f%%" % (scores[1]*100))

fl = open('rnn_loss.txt', 'w')

for loss in hist.history['loss']:
    fl.write(repr(loss) + '\n')

fa = open('rnn_acc.txt', 'w')
for acc in hist.history['acc']:
    fa.write(repr(acc) + '\n')

el = open('rnnev_loss.txt', 'w')

for loss in hist.history['val_loss']:
    el.write(repr(loss) + '\n')

ea = open('rnnev_acc.txt', 'w')
for acc in hist.history['val_acc']:
    ea.write(repr(acc) + '\n')

# demonstrate some model predictions
