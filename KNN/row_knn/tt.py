#coding = utf-8
'''
[hyhong]
2018-5-23
KNN's detailed implementation
'''
import operator
import numpy as np
import pickle
from keras.utils import np_utils

test_class = {}

def distance_count(test_x, train_x, train_y, k):
    '''
    single example in test data
    use Euclidean distance formula and Gauss funtion:
    '''
    flag = 0
    predict_tmp = np.zeros((1, 31))
    for test_x_0 in test_x:
        sq_diff_arr = (test_x_0 - train_x) ** 2
        sq_diff_list = np.zeros((train_x.shape[0], 1))
        for i in range(0, sq_diff_arr.shape[0]):
            sq_diff_sum = 0
            for j in  range(0, sq_diff_arr.shape[1]):
                sq_diff_sum = sq_diff_sum + sq_diff_arr[i][j] * np.exp(-((test_x_0[j] -train_x[i][j])**2))
            sq_diff_list[i] = sq_diff_sum
        #sq_diff_sum = ((test_x_0 - train_x) ** 2).sum(axis = 1)
        distances = sq_diff_list ** 2
        output = open('Euclidean_distance_with_Gauss' + '.csv', 'a')
        for item in distances:
            output.write(str(item) + '\n')
        #if argsort's object is matrix, pay attention to axis; if it is list then ignores it.
        sort_dist = distances.argsort(axis=0)
        _tmp = np.zeros((1, 31))
        for i in range(0, k):
            label = train_y[sort_dist[i]]
            #print('label是——————' + str(label) + '\n')
            _tmp = _tmp + np_utils.to_categorical(label,31)
        if flag == 0:
            predict_tmp = _tmp
        else:
            predict_tmp = np.r_[predict_tmp,_tmp]
        flag = 1
        print(predict_tmp)
    return predict_tmp

def accuracy(predict_y, real_y, f_str):
    count = 0
    total = len(predict_y)
    f = open(f_str, "a")
    for i in len(predict_y):
        f.write(str(predict_y[i]) + ',' +str(real_y[i]) + '\n')
        if predict_y[i] == real_y[i]:
            count = count + 1
    f.close()
    acc = float(count / total)
    return acc

if __name__ == '__main__':
    #import test data
    #neighbors = range(5, 30, 2)
    neighbors = [2]
    with open('../data/smp_test_x_bow.pkl', 'rb') as f:
        test_x = pickle.load(f)
    with open('../data/smp_test_y.pkl', 'rb') as f:
        test_y = pickle.load(f)
    test_matrix = np.zeros((len(test_y), 31))
    for k in neighbors:
        for i in range(0, 2):
            print('running ' + str(k) + '_' + str(i) + ' train....')
            f_str = 'KNN_pre_' + str(k) + '_' + str(i) + '.csv'
            train_x = np.load('../data/' + str(i) + 'train_x_fold.npy')
            train_y = np.load('../data/' + str(i) + 'train_y_fold.npy')
            val_x = np.load('../data/' + str(i) + 'val_x_fold.npy')
            val_y = np.load('../data/' + str(i) + 'val_y_fold.npy')
            valid_matrix = distance_count(val_x, train_x, train_y, k)
            valid_class = np.argmax(valid_matrix, axis=1)
            print('这是valid的prediction\n' + str(valid_class))
            test_matrix_tmp = distance_count(test_x, train_x, train_y, k)
            test_matrix = test_matrix + test_matrix_tmp
            test_class = np.argmax(test_matrix, axis=1)
            print('这是valid的prediction\n' + str(valid_class))
            valid_acc = accuracy(valid_class, val_y, f_str)
            test_acc = accuracy(test_class, test_y, f_str)
            fo = open("KNN.csv", "a")
            fo.write(str(k) + ',' + str(i) + str(valid_acc) + ',' + str(test_acc) + '\n')
