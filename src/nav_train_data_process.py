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
    """
    该函数用来对原始的净值数据进行处理，其中 前90天的数据/当天数据的比值 作为训练用的特征，后30天数据的最大值和最小值/当天数据
    的比值作为之后波动率预测的标签
    :return:
    """
    # 实例化配置文件
    config = Config()
    work_date_list = config.work_date_list
    name_list = config.name_list
    # 读取ODS_MDS.NAV文件
    train_data = pd.read_csv(config.original_nav_csv, encoding="gb18030", usecols=["SYMBOL", "PUBLISHDATE", "NAV1"])
    # 将‘SYMBOL’(基金名)，‘PUBLISHDATE’(发布时间)，‘NAV1’(基金净值)三列
    # 初始化时间序列
    work_date_list = list(train_data.PUBLISHDATE.unique())
    work_date_list.sort()
    # 初始化并清空记录异常的log.txt文件
    with open("log.txt", "w") as f:
        f.seek(0)
        f.truncate()
    # 开始对我们选取的基金数据进行处理
    train = pd.DataFrame()
    for name in name_list:
        try:
            train = train.append(each_fund_process(name, train_data, work_date_list))
        except:
            # 出现异常则记录出现异常的地方
            with open("log.txt", "a+") as f:
                f.write("wrong file:" + str(name) + " \n")
    train.to_csv("../data/processed_data/processed_data.csv",index=False)


def each_fund_process(name, train_data, work_date_list):
    # 选取其中一支基金（‘SYMBOL’为name的）
    train_data = train_data[train_data["SYMBOL"] == name][["PUBLISHDATE", "NAV1"]]
    # 构建时间序列表并按照时间序列排序
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
    # 填充缺失值
    train_data = model_insert(train_data, name)
    # 初始化用来输出结果的dict
    train = {}
    for num in range(train_days):
        train[num] = []
    #‘train’和‘test’列分别用来记录某一条数据中前90天中有多少条真实数据和后30天中有多少条真实数据
    train["train"] = []
    train["test"] = []
    # ‘max_retio’和‘min_ratio’列分别用来记录 后30天中最大值/当天 的比值和 最小值/当天 的比值
    train["max_ratio"] = []
    train["min_ratio"] = []
    # 去掉开始90议案和最后30天的数据，之后对其中每一天的数据进行处理
    for index in range(train_days, len(train_data) - test_days):
        # 取121天的数据进行处理
        data = train_data[index - train_days:index + test_days + 1]
        data.reset_index(drop=True, inplace=True)
        # 前90天的数据
        train_temp = data[0:train_days]
        # 后30天的数据
        pre_temp = data[train_days + 1:train_days + test_days + 1]
        # 当天的数据
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
    train_data['SYMBOL'] = name
    # 输出
    print(str(name) + "  is  Over!!!!")
    return train_data


def model_insert(fund_nav, name):
    """
    利用RandomForest模型进行预测和填充缺失值
    :param fund_nav:
    :param name:
    :return:
    """
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
    # 画图及输出
    fig = plt.figure()
    plt.plot(train_index, target, "b*")
    plt.plot(test_index, result, "ro")
    # fig.show()
    fig.savefig("../data/processed_data/" + str(name) + "__processed_data.png")
    plt.close(fig)
    return fund_nav

