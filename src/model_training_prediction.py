from config import Config
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from os import path
import random

label_name = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
score_dict = {"p5": [], "p10": [], "p15": [], "p-5": [], "p-10": [], "p-15": [], "num": []}
name_list = Config.name_list
fund_type = ['shares', 'bond']
col_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18',
            '19', '20',
            '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38',
            '39', '40', '41', '42', '43', '44', '45', '46',
            '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64',
            '65', '66', '67', '68', '69', '70', '71', '72', '73',
            '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89'
            ]


def train_test_split():
    shares = pd.read_csv("../data/processed_data/shares_fund_data.csv")
    bond = pd.read_csv("../data/processed_data/bond_fund_data.csv")
    shares_index = shares.SYMBOL.tolist()
    bond_index = bond.SYMBOL.tolist()
    shares_index = list(set(shares_index))
    bond_index = list(set(bond_index))
    shares_test_index = random.sample(shares_index, int(0.3 * len(shares_index)))
    bond_test_index = random.sample(bond_index, int(0.3 * len(bond_index)))
    shares_train_index = [num for num in shares_index if num not in shares_test_index]
    bond_train_index = [num for num in bond_index if num not in bond_test_index]
    shares_train = shares[shares['SYMBOL'].isin(shares_train_index)]
    shares_test = shares[shares['SYMBOL'].isin(shares_test_index)]
    bond_train = bond[bond['SYMBOL'].isin(bond_train_index)]
    bond_test = bond[bond['SYMBOL'].isin(bond_test_index)]
    shares_train.to_csv("../data/processed_data/shares_train.csv", index=False)
    shares_test.to_csv("../data/processed_data/shares_test.csv", index=False)
    bond_train.to_csv("../data/processed_data/bond_train.csv", index=False)
    bond_test.to_csv("../data/processed_data/bond_test.csv", index=False)


def model_training():
    """
    训练模型并持久化
    :return:
    """
    # 遍历两种基金(股票型基金和债券型基金)
    for name in fund_type:
        # 读取该种基金的数据表
        data = pd.read_csv("../data/processed_data/" + name + '_train.csv')
        # 为了提高模型的准确性和合理性，选取‘train’（前90天数据中真实数据的数量）和‘test’（后30天数据中真实数据的数量）符合要求的数据
        data = data[(data["train"] >= 80) & (data["test"] >= 20)]
        # # 为了方便处理，重命名dataframe的列名
        # for num in range(len(data.columns)):
        #     data.rename(columns={str(num): num}, inplace=True)
        # 初始化训练集和测试集
        train = data.loc[:, col_name]
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
    for name in fund_type:
        data = pd.read_csv('../data/processed_data/' + name + '_test.csv')
        data = data[(data["train"] >= 80) & (data["test"] >= 20)]
        # 初始化训练集和测试集
        test = data.loc[:, col_name]
        target = data.loc[:, label_name]
        # test = np.array(np.var(test.iloc[:, 0:90], axis=1)).reshape(-1, 1)
        # target = pd.read_csv('../data/processed_data/' + name + "_target_data.csv")
        for label in label_name:
            clf = joblib.load('../data/model/' + name + '_' + label + "_train_model.m")
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
