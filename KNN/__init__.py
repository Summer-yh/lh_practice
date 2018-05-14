'''
[hyhong]
2018-5-13
'''
#coding = utf-8
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

#max number of word
maxlen = 20
#each word's dimension
word_dim = 50

def Cross_validation(index, n_nbor, x_train, y_train, x_dev, y_dev, p, weights, leaf_size):
    print('model is running...\n')
    model = KNeighborsClassifier(n_neighbors = n_nbor, weights = weights, leaf_size = leaf_size, p = p)
    model.fit(x_train, y_train)
