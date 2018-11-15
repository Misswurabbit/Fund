from config import Config
import pandas as pd
import numpy as np
from xgboost.sklearn import XGBRegressor, XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
from os import path

label_name = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
score_dict = {"p5": [], "p10": [], "p15": [], "p-5": [], "p-10": [], "p-15": [], "num": []}
name_list = Config.name_list
fund_type = ['shares_fund', 'mixed_fund', 'bond_fund', 'QDII_fund']


def model_training():
    for name in fund_type:
        data = pd.read_csv("../data/processed_data/" + name + "_data.csv")
        data = data[(data["train"] >= 80) & (data["test"] >= 20)]
        for num in range(len(data.columns)):
            data.rename(columns={str(num): num}, inplace=True)
        train = np.array(np.var(data.iloc[:, 0:90], axis=1)).reshape(-1, 1)
        # train = data.loc[:, [num for num in range(90)]]
        # train.sort_index(axis=1, inplace=True)
        target = data.loc[:, label_name]
        for label in label_name:
            clf = RandomForestClassifier()
            clf.fit(train, target.loc[:, label])
            # print('../../data/model/' + name + '_' + label + "_train_model.m")
            joblib.dump(clf, '../data/model/' + name + '_' + label + "_train_model.m")
        print(name + " model is over")


def model_prediction():
    for name in ['bond', 'shares']:
        test = pd.read_csv('../data/processed_data/' + name + "_test_data.csv")
        test = np.array(np.var(test.iloc[:, 0:90], axis=1)).reshape(-1, 1)
        answer = pd.read_csv('../data/processed_data/' + name + "_target_data.csv")
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
            score = accuracy_score(answer.loc[:, label], result)
            print(score)


model_prediction()
