from config import Config
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from os import path

label_name = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
score_dict = {"p5": [], "p10": [], "p15": [], "p-5": [], "p-10": [], "p-15": [], "num": []}
name_list = Config.name_list
fund_type = ['shares_fund', 'mixed_fund', 'bond_fund', 'QDII_fund']


def model_training():
    """
    训练模型并持久化
    :return:
    """
    # 遍历两种基金(股票型基金和债券型基金)
    for name in fund_type:
        # 读取该种基金的数据表
        data = pd.read_csv("../data/processed_data/" + name + "_data.csv")
        # 为了提高模型的准确性和合理性，选取‘train’（前90天数据中真实数据的数量）和‘test’（后30天数据中真实数据的数量）符合要求的数据
        data = data[(data["train"] >= 80) & (data["test"] >= 20)]
        # # 为了方便处理，重命名dataframe的列名
        # for num in range(len(data.columns)):
        #     data.rename(columns={str(num): num}, inplace=True)
        # 初始化训练集和测试集
        train = np.array(np.var(data.loc[:, [num for num in range(90)]], axis=1)).reshape(-1, 1)
        target = data.loc[:, label_name]
        for label in label_name:
            # 模型初始化
            clf = RandomForestClassifier()
            # 模型训练
            clf.fit(train, target.loc[:, label])
            # 保存模型
            joblib.dump(clf, '../data/model/' + name + '_' + label + "_train_model.m")
        print(name + " model is over")


def model_prediction():
    """
    用训练好的模型进行预测
    :return:
    """
    with open('accurancy.txt', 'w') as f:
        f.seek(0)
        f.truncate()
    for name in ['bond', 'shares']:
        test = pd.read_csv('../data/processed_data/' + name + "_test_data.csv")
        test = np.array(np.var(test.iloc[:, 0:90], axis=1)).reshape(-1, 1)
        target = pd.read_csv('../data/processed_data/' + name + "_target_data.csv")
        for label in label_name:
            clf = joblib.load('../data/model/' + name + '_fund_' + label + "_train_model.m")
            proba = clf.predict_proba(test)
            result = []
            if proba.shape[1] == 2:
                proba = [[temp[0], temp[1]] for temp in proba]
            else:
                if clf.predict(test)[0]:
                    proba = [[0, 1] for temp in range(len(proba))]
                else:
                    proba = [[1, 0] for temp in range(len(proba))]
            for temp in proba:
                if temp[1] >= 0.9:
                    result.append(True)
                else:
                    result.append(False)
            result = pd.Series(result)
            score = accuracy_score(target.loc[:, label], result)
            with open('accurancy.txt', 'a+') as f:
                f.write(name + "  " + label + "    accurancy:" + str(score) + '\n')
