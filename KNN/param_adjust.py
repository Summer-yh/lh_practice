#coding:utf-8
'''
[hyhong]
2018-5-13
'''
import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

#max number of word
maxlen = 20
#each word's dimension
word_dim = 50

def get_test_acc(y_predict, y_test):
    '''
    get correspondence between y's prediction and y's true value
    '''
    count = 0
    for i in range(len(y_predict)):
        '''
        argsort() means ascending order，[-1] means get the last one which is the maximum
        '''
        predict_label = np.array(y_predict[i]).argsort()[-1]
        if predict_label == y_test[i]:
            count += 1
        test_acc = float(count / len(y_test))
    return test_acc

if __name__ == '__main__':
    #set parameters
    param_list = {
            'n_neighbors' : range(3, 30, 2),
            'p' : [1, 2],
            'leaf_size' : range(15, 50, 2),
            'weights' : ['uniform', 'distance']
            }
    #import test data
    with open('data/smp_test_x_bow.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('data/smp_test_y.pkl', 'rb') as f:
        y_test = pickle.load(f)
    t = 0
    knn = KNeighborsClassifier()
    knn_search = GridSearchCV(knn, param_list)
    predict = 0
    cross = 0
    for i in range(0, 5):
        #import train datas and validation datas
        x_train = np.load('data/' + str(i) + 'train_x_fold.npy')
        y_train = np.load('data/' + str(i) + 'train_y_fold.npy')
        x_val = np.load('data/' + str(i) + 'val_x_fold.npy')
        y_val = np.load('data/' + str(i) + 'val_y_fold.npy')
        print(2312321432423)
        knn_search.fit(x_train, y_train)
        print('fit is done\n')
        val_score = knn_search.score(x_val, y_val)
        print('i------acc is '  + str(val_acc))
        print(knn_search.cv_result_)
        cross += val_score
        tmp = knn_search.predict_proba(x_test)
        predict += tmp
    cross_acc = cross / 5
    test_acc = get_test_acc(predict, y_test)
    f = open("KNN.csv", "a")
    f.write(str(cross_acc) + ',' + str(test_acc) + ',' +'\n')
