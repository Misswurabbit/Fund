import pandas as pd
from os import path

fund_type = ['bond','shares']

def data_process():
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
        target['p5'] = target.max_ratio > 1.05
        target['p10'] = target.max_ratio > 1.1
        target['p15'] = target.max_ratio > 1.15
        target['p-5'] = target.min_ratio < 0.95
        target['p-10'] = target.min_ratio < 0.9
        target['p-15'] = target.min_ratio < 0.85
        target = target[['p5', 'p10', 'p15', 'p-5', 'p-10', 'p-15']]
        train.to_csv("../../data/processed_data/"+type+"_test_data.csv",index=False)
        target.to_csv("../../data/processed_data/"+type+"_target_data.csv", index=False)
        print("Over")
