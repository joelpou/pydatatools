#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from time import time

(fig, subplots) = plt.subplots(1, 4, figsize=(15, 8))
perplexities = [5, 30, 50, 100]
# siftfeatures = np.genfromtxt('.\sift-indexes\sift_index_23.csv', delimiter=',', usecols=np.arange(1,513), dtype=np.float32) # , invalid_raise=False
# siftfeatures = np.genfromtxt('.\sift-indexes\sift_index_23.csv', delimiter=',', usecols=np.arange(513,1025), dtype=np.float32) # , invalid_raise=False
# siftfeatures = np.genfromtxt('.\sift-indexes\sift_index_24.csv', delimiter=',', usecols=np.arange(1025, 1537), dtype=np.float32) # , invalid_raise=False
siftfeatures = np.genfromtxt('.\lbpfeats\lbpfeatures.csv', delimiter=',', usecols=np.arange(0,99120), dtype=np.float32) # , invalid_raise=False
siftlabels = np.genfromtxt('.\lbpfeats\lbpfeatures.csv', delimiter=',', usecols=(99120), dtype=np.float32)
red = siftlabels == 1
blue = siftlabels == 0
for i, perplexity in enumerate(perplexities):
    ax = subplots[i]
    t0 = time()
    tsne = TSNE(n_components=2, init='random',
                         random_state=0, perplexity=perplexity)
    Y = tsne.fit_transform(siftfeatures)
    t1 = time()
    print("sift features, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))
    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[blue, 0], Y[blue, 1], c="b")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')
plt.show()