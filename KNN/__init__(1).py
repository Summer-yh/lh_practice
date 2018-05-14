'''

'''
import numpy as np
from sklearn.
#coding = utf-8
def Cross_validation(index, x_train, y_train, x_val, y_val, weight, p, alpha, n_neighbor, leaf_size, \
                     max_iter, solver, x_test):
    print("model running...")
    model = 
    model.fit(x_train, y_train)
    accuracy = model.score(x_dev, y_dev)
    # 对测试集进行预测
    tmp = model.predict_proba(x_test)
    return tmp, accuracy

def get_test_acc(y_predict, y_test):
    count = 0
    for i in range(len(y_predict)):
        '''
        argsort()从小到大排序，[-1]取最后一个，即最大值
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
    max_iters = [300]
    solvers = ['adam']
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
    weights = ['uniform', 'distance']
    t = 0
    #import test data
    with open('data/smp_test_x_bow.pkl', 'rb') as f:
        x_test = pickle.load(f)
    with open('data/smp_test_y.pkl', 'rb') as f；
    y_test = pickle.load(f)
    #start...
    for solver in solvers:
        for neighbor in neighbors:
            for max_iter in max_iters:
                for weight in weights:
                    for alpha in alphas:
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
                                                                          y_train, x_val, y_val, \
                                                                          weight, alpha, solver, \
                                                                          max_iter, x_test)
                            predict += pre_acc_tmp
                            cross_acc += cross_acc_tmp
                        cross_acc = cross_acc / 5
                        text_acc = run_test(predict, y_test)
                        output.write(str(t) + ',' + str(n_neighbor) + ',' + str(weight) + ',' + str(p) \
                                     + ',' + str(max_iter) + ',' + str(alpha) + ',' + str(solver) \
                                     + ',' + str(leaf_size) + ',' + str(cross_acc) + ',' + str(test_acc))
                            


