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

def Cross_validation(index, n_nbor, x_train, y_train, x_dev, y_dev, p, weight, leaf_size, x_test):
    '''
    cross validation
    return validation's accuracy and test prediction's ratio redult
    '''
    print('model is running...\n')
    model = KNeighborsClassifier(n_neighbors = n_nbor, weights = weight, leaf_size = leaf_size, p = p)
    model.fit(x_train, y_train)
    accuracy = model.score(x_val, y_val)
    # prediction for test data
    tmp = model.predict_proba(x_test)
    return tmp, accuracy

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

if __name__ = 'main':
    output = open('KNN.cvs', 'a')
    output.write('index,n_neighbors,weight,p,max_iter,alpha,solver,leaf_size,cross_acc,test_acc\n')
    #set parameters
    n_neighbors = [5, 10, 15, 20, 25]
    p = [1, 2]
    leaf_sizes = [20, 25, 30, 35, 40, 50]
    weights = ['uniform', 'distance']
    t = 0
    #import test data
    with open('data/smp_test_x_bow.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('data/smp_test_y.pkl', 'rb') as f；
    y_test = pickle.load(f)
    #start...
    for neighbor in neighbors:
        for leaf_size in leaf_sizes:
            for weight in weights:
                for p_0 in p:
                    t = t + 1
                    #predict is list cox predict_proba() return list of prediction
                    predict = [0] * 31
                    cross_acc = 0
                    for i in range(1, 5):
                        index = str(t) + '.' + str(i)
                        print('running ' + index + ' ....')
                        #import train datas and validation datas
                        x_train = np.load('data/' + str(i) + 'train_x_fold.npy')
                        y_train = np.load('data/' + str(i) + 'train_y_fold.npy')
                        x_val = np.load('data/' + str(i) + 'val_x_fold.npy')
                        y_val = np.load('data/' + str(i) + 'val_y_fold.npy')
                        pre_acc_tmp, cross_acc_tmp = Cross_validation(index, neighbor, x_train, \
                                                                      y_train, x_val, y_val, weight, \
                                                                      p_0, leaf_size, x_test)
                        predict += pre_acc_tmp
                        cross_acc += cross_acc_tmp
                    cross_acc = cross_acc / 5
                    text_acc = run_test(predict, y_test)
                    output.write(str(t) + ',' + str(n_neighbor) + ',' + str(weight) + ',' + str(p) \
                                 + ',' + str(max_iter) + ',' + str(alpha) + ',' + str(solver) \
                                 + ',' + str(leaf_size) + ',' + str(cross_acc) + ',' + str(test_acc))
                            


