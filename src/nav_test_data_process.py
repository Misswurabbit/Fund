import pandas as pd
from os import path
from config import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xlrd
fund_type = ['bond','shares']

def nav_test_data():
    for type in fund_type:
        fund_path = path.abspath(path.join(__file__, "../../data/original/"+type+"_fund_2018.csv"))
        data = pd.read_csv(fund_path, encoding="gb18030")
        data = data.iloc[:, 1:]
        data.columns = [num for num in range(len(data.columns))]
        for num in range(len(data.columns)):
            data[num] = pd.to_numeric(data[num], errors='coerce')
        data.dropna(inplace=True)
        base = data.iloc[:, 90]
        for num in range(len(data.columns)):
            data[num] = data[num] / base
        train = data.iloc[:, 0:90]
        target = data.iloc[:, 91:121]
        max_ratio = []
        min_ratio = []
        for value in target.values:
            max_ratio.append(max(value))
            min_ratio.append(min(value))
        target["max_ratio"] = max_ratio
        target["min_ratio"] = min_ratio
        target['p5'] = target.max_ratio > 1.02
        target['p10'] = target.max_ratio > 1.06
        target['p15'] = target.max_ratio > 1.1
        target['p-5'] = target.min_ratio < 0.98
        target['p-10'] = target.min_ratio < 0.94
        target['p-15'] = target.min_ratio < 0.9
        target = target[['p5', 'p10', 'p15', 'p-5', 'p-10', 'p-15']]
        train.to_csv("../../data/processed_data/"+type+"_test_data.csv",index=False)
        target.to_csv("../../data/processed_data/"+type+"_target_data.csv", index=False)
        print("Over")
