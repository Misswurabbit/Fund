from config import Config
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# import warnings
# warnings.filterwarnings("ignore")
train_days = 90
test_days = 30
total_days = train_days + test_days


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
    #each_fund_process(train_data["SYMBOL"].iloc[0], train_data, work_date_list)
    for name in name_list:
        try:
            each_fund_process(name, train_data, work_date_list)
            print("Over")
        except:
            with open("log.txt", "a+") as f:
                f.seek(0)
                f.truncate()
                f.write("wrong file:"+str(name)+" \n")


def each_fund_process(name, train_data, work_date_list):
    train_data = train_data[train_data["SYMBOL"] == name][["PUBLISHDATE", "NAV1"]]
    train_data.sort_values(by="PUBLISHDATE", ascending=True, inplace=True)
    train_data["SYMBOL"] = [name for num in range(len(train_data))]
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
    train_data.fillna({'real':False},inplace= True)
    train_data = model_insert(train_data, name)
    train_data['max'] = train_data.rolling(window=test_days).max().NAV1
    train_data['min'] = train_data.rolling(window=test_days).min().NAV1
    train_data['train'] = train_data.real.rolling(window=train_days).sum()
    train_data['test'] = train_data.real.rolling(window=test_days).sum()
    train_data['max'] = train_data['max'].shift(-test_days)
    train_data['min'] = train_data['min'].shift(-test_days)
    train_data['train'] = train_data['train'].shift(-train_days)
    train_data['test'] = train_data['test'].shift(-total_days)
    train_data['max_ratio'] = train_data.apply(divide_max, axis=1)
    train_data['min_ratio'] = train_data.apply(divide_min, axis=1)
    train_data['p5'] = train_data.max_ratio > 1.05
    train_data['p10'] = train_data.max_ratio > 1.1
    train_data['p15'] = train_data.max_ratio > 1.15
    train_data['p-5'] = train_data.min_ratio < 0.95
    train_data['p-10'] = train_data.min_ratio < 0.9
    train_data['p-15'] = train_data.min_ratio < 0.85
    sample_list = []
    for i in range(train_data.shape[0] - total_days):
        sample_list.append(train_data.iloc[i:i + train_days, 0].tolist())
    feature = pd.DataFrame(data=sample_list)
    feature.index = train_data.index.tolist()[:train_data.shape[0] - total_days]
    train = pd.concat([feature, train_data.iloc[:train_data.shape[0] - total_days].loc[:,['train','test','p5','p10','p15','p-5','p-10','p-15']]], axis=1)
    for i in range(90):
        train[i] = train.apply(divide, args=[i], axis=1)
    train.to_csv("../data/processed_data/" + str(name) + "__processed_data.csv")
    print(str(name) + "  Over!!!!")


def divide(item, i):
    try:
        return item.iloc[i] / item.iloc[89]
    except:
        return item.iloc[i]

def divide_max(train_data):
    return train_data['max'] / train_data['NAV1']


def divide_min(train_data):
    return train_data['min'] / train_data['NAV1']


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
            fund_nav.loc[str(date),"NAV1"] = result[num]
            num += 1

    fig = plt.figure()
    plt.plot(train_index, target, "b*")
    plt.plot(test_index, result, "ro")
    # fig.show()
    fig.savefig("../data/processed_data/" + str(name) + "__processed_data.png")
    plt.close(fig)
    return fund_nav


data_process()
