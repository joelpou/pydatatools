#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from time import time

t0 = time()

lbpfeatures_real = np.genfromtxt('./lbpfeatures_train_real.csv', delimiter=',', usecols=np.arange(0,99120), dtype=np.float32)
lbpfeatures_spoof = np.genfromtxt('./lbpfeatures_train_attack.csv', delimiter=',', usecols=np.arange(0,99120), dtype=np.float32)
lbpfeatures_all = np.vstack((lbpfeatures_real,lbpfeatures_spoof))

print('data shape is:\n')
print(lbpfeatures_all.shape)

fp = np.memmap('./lbpfeatures_train_pca.bin', dtype='float32', mode='w+', shape=lbpfeatures_all.shape)

fp[:] = lbpfeatures_all[:]

del fp

t1 = time()

print('done in %.2g sec' % (t1-t0))
