#coding = utf-8
'''
[hyhong]
2018-5-23
KNN's detailed implementation
'''
import operator
import numpy as np
import pickle

test_class = {}

def distance_count(test_x, train_x, train_y, k):
    '''
    single example in test data
    use Euclidean distance formula and Gauss funtion
    '''
    
    train_len = train_x.shape[0]
    for test_x_0 in test_x:
        
        sq_diff_arr = (test_x_0 - train_x) ** 2
        sq_diff_list = list()
        for i in range(0, sq_diff_arr.shape[0]):
            sq_diff_sum = 0
            for j in  range(0, sq_diff_arr.shape[1]):
                sq_diff_sum = sq_diff_sum + sq_diff_arr[i][j] * np.exp(-((test_x_0[j] -train_x[i][j])**2))
            sq_diff_list.append(sq_diff_sum)
        #sq_diff_sum = ((test_x_0 - train_x) ** 2).sum(axis = 1)
        distances = np.power(sq_diff_list, 2)
        sort_dist = distances.argsort()
        print('sdsd' + str(distances))
        class_count = {}
        for i in range(0, k):
            label = train_y[sort_dist[i]]
            class_count[label] = class_count.get(label, 0) + 1 
        sort_class = sorted(class_count.items(), key = operator.itemgetter(1), reverse = True)
        return sort_class

def summerize(sort_class):
    for item in sort_class:
        test_class[item.[0]] = test_class.get(item[0], 0) + sort_class[0][0]
    return 

if __name__ == '__main__':
    #import test data
    neighbors = range(5, 30, 2)
    with open('../data/smp_test_x_bow.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('../data/smp_test_y.pkl', 'rb') as f:
        y_test = pickle.load(f)
    for k in neighbors:
        for i in range(0, 5):
            count = 0
            f_str = 'KNN_pre_' + str(k) + '_' + str(i) + '.csv'
            train_x = np.load('../data/' + str(i) + 'train_x_fold.npy')
            train_y = np.load('../data/' + str(i) + 'train_y_fold.npy')
            val_x = np.load('../data/' + str(i) + 'val_x_fold.npy')
            val_y = np.load('../data/' + str(i) + 'val_y_fold.npy')
            for j in range(0, val_x.shape[0]):
                pred_class = distance_count(val_x[j], train_x, train_y, k)
                predict_y = pred_class[0][0]
                f = open(f_str, "a")
                f.write(str(val_y[j]) + ',' +str(predict_y) + '\n')
                if val_y[j] == predict_y:
                    count = count + 1
            acc = count / val_y.shape[0]
            for l in range(0, x_test.shape[0]):
                test_predict_y = distance_count(x_test[l], train_x, train_y, k)
                f = open(f_str, "a")
                f.write(str(val_y[j]) + ',' +str(predict_y) + '\n')
                if val_y[j] == predict_y:
                    count = count + 1
            acc = count / val_y.shape[0]
            f = open("KNN.csv", "a")
            f.write(str(count) + ',' + str(acc) + '\n')
