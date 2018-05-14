#encoding:utf-8
from __future__ import print_function

import pickle
import numpy as np
from keras.utils import np_utils
from sklearn.neural_network import MLPClassifier

# 最大词数
maxlen = 20
# 每个词映射的维度
len_wv = 50


def Cross_validation(index, x_train, y_train, x_dev, y_dev, hidden_layer_size, \
                     max_iter, alpha, solver, x_test):
    print("model running...")
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_size, max_iter=max_iter, alpha=alpha, solver=solver)
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

if __name__ == '__main__':
    output = open("MLP.csv", "a")
    output.write("acc,index,hidden_layer_size,,max_iter,alpha,solver\n")
    output2 = open("MLP_avg_acc.csv", "a")
    output2.write("index,train_avg_acc,test_acc\n")
    hidden_layer_sizes = [[100, 100], [100, 150], [100, 200], [150, 100], [200, 100],
                          [150, 150], [200, 150], [200, 200]]
    max_iters = [300]
    solvers = ["adam"]
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.5]
    t = 0
    with open("SMP2017/data/smp_test_x_bow.pkl","rb") as f:
        x_test= pickle.load(f)
    with open("SMP2017/data/smp_test_y.pkl", "rb") as f:
        y_test= pickle.load(f)
    for solver in solvers:
        for hidden_layer_size in hidden_layer_sizes:
            for max_iter in max_iters:
                for alpha in alphas:
                    t = t + 1
                    predict = list()
                    acc = 0
                    for i in range(0, 5):
                        index = str(t) + '.' + str(i)
                        print("running"+index+'....')
                        x_train = np.load("SMP2017/data/" + str(i) +
                                          "train_x_fold.npy")
                        x_dev = np.load("SMP2017/data/" + str(i) +
                                          "val_x_fold.npy")
                        # x_test = np.load("H:/research/data/SMP2017/smp2017_5fold/" + str(i) +
                        #                  "develop_x_test_smp_bow.npy")
                        # y_test = np.load("H:/research/data/SMP2017/smp2017_5fold/" + str(i) +
                        #                   "develop_y_test_smp_bow.npy")
                        y_train = np.load("SMP2017/data/" + str(i) +
                                          "train_y_fold.npy")
                        y_dev = np.load("SMP2017/data/" + str(i) +
                                          "val_y_fold.npy")


                        # y_dev = np_utils.to_categorical(y_dev, 31)  # 必须使用固定格式表示标签
                        # y_train = np_utils.to_categorical(y_train, 31)  # 必须使用固定格式表示标签 一共 31分类

                        predict_tmp, acc_tmp = Cross_validation(index, x_train,y_train, x_dev, y_dev, \
                                               hidden_layer_size, max_iter, alpha, solver, x_test)

                        if len(predict) == 0:
                            predict = predict_tmp
                        else:
                            predict += predict_tmp
                        acc += acc_tmp
                        print(acc_tmp)
                        output.write(str(acc_tmp) +',' + str(index) + ',' + str(hidden_layer_size) + \
                                     ',' + str(max_iter) + ',' + str(alpha) + ',' + str(solver) + "\n")
                    avg_acc = acc/5
                    test_acc = get_test_acc(predict, y_test)
                    output2.write(str(t) + ',' + str(avg_acc) + ',' + str(test_acc) + '\n')
