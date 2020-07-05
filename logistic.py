import numpy as np
from random import shuffle
from numpy.linalg import inv
from math import floor, log
import matplotlib.pyplot as plt
import os
import argparse
import pandas as pd

dir = './Data/'

def washData(pathData, pathAnswer='Nothing'):
    # 14个属性+收入属性
    # 数据清洗
    df_x = pd.read_csv(pathData)
    # 在执行清洗之前,合并数据和答案，方便将行数据对应清洗
    if (pathAnswer != 'Nothing'):  # 表示是测试数据，真的有pathAnswer
        df_ans = pd.read_csv(pathAnswer)
        df_x = pd.concat([df_x, df_ans['label']], axis=1)  # 注意训练集里面列名是'income', 这里是'label'
        df_x.rename(columns={'label': 'income'}, inplace=True)  # label -> income
    else:
        df_x['income'] = (df_x['income'] == ' >50K')
    df_x = df_x.replace(' ?', np.nan)  # 将数据中存在'?'的行用NAN替代
    df_x = df_x.dropna()  # 将含有NAN的行删除

    #  修改性别项 和 分离income项
    df_x['sex'] = (df_x['sex'] == 'male')
    data_y = df_x[['income']].astype(np.int64)  # df_x[[]]两重括号才能保持其DataFrame属性, 一重括号data_y变成Series属性
    del df_x['income']

    # 将数据分成数字和非数字 两部分
    object_columns = [col for col in df_x.columns if
                      df_x[col].dtypes == "object"]  # 陷阱：in df.columns可以，in range(df.columns)不行
    no_object_columns = [col for col in df_x.columns if df_x[col].dtypes == 'int64']
    object_data = df_x[object_columns]
    no_object_data = df_x[no_object_columns]
    # set every element in object rows as an attribute
    object_data = pd.get_dummies(object_data)  # 走到这一步其实很多列映射的值都一样
    # 将数字部分和非数字部分都合并起来，还是我们的数据集
    data_x = pd.concat([no_object_data, object_data], axis=1)
    data_x = data_x.astype(np.int64)
    # 数据都变成了一些数字
    data_x = (data_x - data_x.mean()) / data_x.std()

    if pathAnswer == 'Nothing':  # 对比train.csv和test.csv发现如下项对应不了，故train.csv中获取的此元素删掉
        del data_x['native_country_ Holand-Netherlands']
    return data_x.values, data_y.values  # 分别为14列、1列 # 这.values是陷阱啊！！！没有要不得，findParams会取不出数字的

def sigmoid(z):
    z = 1 / (1.0 + np.exp(-z))
    return z

def g_train(X, Y):
    # 我们需要u1, u2, E1, E2来计算 z=w*x+b的w、b
    num = X.shape[0]
    cnt1 = 0
    cnt2 = 0

    sum1 = np.zeros((X.shape[1],))  # (101,)
    sum2 = np.zeros((X.shape[1],))
    for i in range(num):
        if Y[i] == 1:
            sum1 += X[i]
            cnt1 += 1
        else:
            sum2 += X[i]
            cnt2 += 1
    u1 = sum1 / cnt1
    u2 = sum2 / cnt2  # 找到了平均值

    E1 = np.zeros((X.shape[1], X.shape[1]))  # (101, 101)
    E2 = np.zeros((X.shape[1], X.shape[1]))  # (101, 101)
    for i in range(num):
        if Y[i] == 1:
            # E1 += np.dot(X[i] - u1, (X[i] - u1).T)
            E1 += np.dot(np.transpose([X[i] - u1]), [X[i] - u1])
        else:
            # E2 += np.dot(X[i] - u2, (X[i] - u2).T)
            E2 += np.dot(np.transpose([X[i] - u2]), [X[i] - u2])

    E1 = E1 / float(cnt1)
    E2 = E2 / float(cnt2)
    E = E1 * (float(cnt1) / num) + E2 * (float(cnt2) / num)

    #print ('findParams_U1', u1.shape, u1)
    #print ('findParams_U2', u2.shape, u2)
    #print ('findParams_E', E.shape, E)
    return u1, u2, E, cnt1, cnt2

trainX, trainY = washData(dir + '/train.csv')  # trainX是DataFrame(30162, 101)  (30162,)
testX, testY = washData(dir + '/test.csv', dir + '/correct_answer.csv')  # (15060, 101) (15060,)
def g_pridict(X, Y, u1, u2, E, N1, N2):
    E_inv = inv(E)  # 居然碰到奇异矩阵的问题
    w = np.dot((u1 - u2), E_inv)
    b = (-0.5) * np.dot(np.dot(u1.T, E_inv), u1) + (0.5) * np.dot(np.dot(u2.T, E_inv), u2) + np.log(float(N1) / N2)
    z = np.dot(w, X.T) + b
    y = sigmoid(z)
    print('w=',w)
    print('b=',b)
    np.savetxt(dir + '/w_value.csv', w)

    cnt1 = 0
    cnt2 = 0
    y = np.around(y)
    for i in range(Y.shape[0]):
        if y[i] == Y[i]:
            cnt1 += 1
        else:
            cnt2 += 1

    print('逻辑回归测试数据共', Y.shape[0], '个，判断正确', cnt1, '个，判断错误', cnt2, '个')
    print('准确率:', float(cnt1) / Y.shape[0] * 100, '%')
    return y

u1, u2, E, N1, N2 = g_train(trainX, trainY)
my_ans = g_pridict(testX, testY, u1, u2, E, N1, N2)
np.savetxt(dir + '/my_ans_1.csv', my_ans)
