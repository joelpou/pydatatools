#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.decomposition import IncrementalPCA

n_batches = 10

m = 3583

n = 99120

X_mm = np.memmap('./lbpfeatures_train_pca.bin', dtype='float32', mode='r', shape=(m, n))

batch_size = m // n_batches

inc_pca = IncrementalPCA(n_components=355, batch_size=batch_size)

inc_pca.fit(X_mm)

print("first eigenvector:\n")

print(inc_pca.components_.T[:, 0])

print("transformation matrix shape:\n")

print(inc_pca.components_.shape)

print("explained_variance_ratio_:\n")

print(np.cumsum(inc_pca.explained_variance_ratio_)[-1])

np.savetxt('./AS_PCA_T.csv', inc_pca.components_, delimiter=',')
