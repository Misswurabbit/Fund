from config import Config
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor, XGBClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import warnings
# warnings.filterwarnings("ignore")
train_days = 90
test_days = 30
total_days = train_days + test_days
labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]


def data_process():
    # 实例化配置文件
    config = Config()
    work_date_list = config.work_date_list
    name_list = config.name_list
    # 读取ODS_MDS.NAV文件
    original_nav_data = pd.read_csv(config.original_nav_csv, encoding="gb18030")
    train_data = original_nav_data[["SYMBOL", "PUBLISHDATE", "NAV1"]]
    work_date_list = list(train_data.PUBLISHDATE.unique())
    work_date_list.sort()
    with open("log.txt", "w") as f:
        f.seek(0)
        f.truncate()
    for name in name_list:
        try:
            each_fund_process(name, train_data, work_date_list)
            print("Over")
        except:
            with open("log.txt", "a+") as f:
                f.write("wrong file:" + str(name) + " \n")


def each_fund_process(name, train_data, work_date_list):
    train_data = train_data[train_data["SYMBOL"] == name][["PUBLISHDATE", "NAV1"]]
    train_data.sort_values(by="PUBLISHDATE", ascending=True, inplace=True)
    first_date = train_data.iloc[0, 0]
    last_data = train_data.iloc[-1, 0]
    first_date_pos = work_date_list.index(first_date)
    last_data_pos = work_date_list.index(last_data)
    work_date_list = work_date_list[first_date_pos:last_data_pos + 1]
    date = pd.DataFrame(data=work_date_list, columns=['PUBLISHDATE'])
    train_data['real'] = True
    train_data = train_data.merge(date, how='right', on=['PUBLISHDATE'])
    train_data.sort_values(by="PUBLISHDATE", ascending=True, inplace=True)
    train_data.set_index(keys='PUBLISHDATE', inplace=True)
    train_data.fillna({'real': False}, inplace=True)
    train_data = model_insert(train_data, name)
    train_data["SYMBOL"] = [name for num in range(len(train_data))]
    train = {}
    for num in range(train_days):
        train[num] = []
    train["train"] = []
    train["test"] = []
    train["max_ratio"] = []
    train["min_ratio"] = []
    for index in range(train_days, len(train_data) - test_days):
        data = train_data[index - train_days:index + test_days + 1]
        data.reset_index(drop=True, inplace=True)
        train_temp = data[0:train_days]
        pre_temp = data[train_days + 1:train_days + test_days + 1]
        base = data.loc[90, 'NAV1']
        train["train"].append(train_temp.real.sum())
        train["test"].append(pre_temp.real.sum())
        train["max_ratio"].append(pre_temp["NAV1"].max() / base)
        train["min_ratio"].append(pre_temp["NAV1"].min() / base)
        num = 0
        for temp in train_temp.values:
            train[num].append(temp[0] / base)
            num += 1
    train_data = pd.DataFrame(train)
    train_data['p5'] = train_data.max_ratio > 1.05
    train_data['p10'] = train_data.max_ratio > 1.1
    train_data['p15'] = train_data.max_ratio > 1.15
    train_data['p-5'] = train_data.min_ratio < 0.95
    train_data['p-10'] = train_data.min_ratio < 0.9
    train_data['p-15'] = train_data.min_ratio < 0.85
    train_data.to_csv("../data/processed_data/" + str(name) + "__processed_data.csv", index=False)
    print(str(name) + "  is  Over!!!!")


def model_insert(fund_nav, name):
    date_index = [pd.to_datetime(date) for date in fund_nav.index]
    origin = date_index[0]
    train_index = []
    target = []
    test_index = []
    for date in date_index:
        coordinate_date = (date - origin).days
        if pd.isna(fund_nav.loc[str(date)]).at["NAV1"]:
            test_index.append(coordinate_date)
        else:
            train_index.append(coordinate_date)
            target.append(fund_nav.loc[str(date)].at["NAV1"])
    clf = RandomForestRegressor()
    clf.fit(np.array(train_index).reshape(-1, 1), target)
    result = clf.predict(np.array(test_index).reshape(-1, 1))
    num = 0
    for date in date_index:
        if pd.isna(fund_nav.loc[str(date)]).at["NAV1"]:
            fund_nav.loc[str(date), "NAV1"] = result[num]
            num += 1

    fig = plt.figure()
    plt.plot(train_index, target, "b*")
    plt.plot(test_index, result, "ro")
    # fig.show()
    fig.savefig("../data/processed_data/" + str(name) + "__processed_data.png")
    plt.close(fig)
    return fund_nav


data_process()
