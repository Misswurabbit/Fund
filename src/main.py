import pandas as pd
from os import path
from config import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# original_bsheet_csv = path.abspath(path.join(__file__, "../../data/original/ODS_MDS.BSHEET.csv"))
# file = pd.read_csv(original_bsheet_csv, encoding="gb18030")
# add_data = file.loc[:, ["SYMBOL", "BSHEET10", "BSHEET11", "BSHEET55", "BSHEET56"]]
# add_data.reset_index(drop=True, inplace=True)
# add_data["SYMBOL"] = pd.to_numeric(add_data["SYMBOL"], errors="coerce")
# add_data.dropna(inplace=True)
# labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
# combine = []
# for name in [1]:
#     data = pd.read_csv("../data/processed_data/" + str(name) + "__processed_data.csv")
#     data = data[(data["train"] >= 45) & (data["test"] >= 15)]
#     data["SYMBOL"] = [name for num in range(len(data))]
#     combine.append(data)
#     # add_data = temp[temp["SYMBOL"] == name]
# total = combine[0]
# # for num in range(1, len(combine)):
# #     total = total.append(combine[num])
# # total.to_csv("total.csv")
# total = total.merge(add_data, on="SYMBOL", how="outer")
# total.dropna(inplace=True)
# total.reset_index(drop=True,inplace=True)
# #total.drop()
# # total.to_csv("__total__.csv", index=False)
total = pd.read_csv("__total__.csv")
index = ["BSHEET10", "BSHEET11", "BSHEET55", "BSHEET56"]
for num in  range(89):
    index.append(str(num))
train = total.loc[:, index]
target = total.loc[:, labelname].astype(bool)
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.3)
model = RandomForestClassifier()
model.fit(x_train,y_train)
result = model.predict(x_test)
have = accuracy_score(y_test,result)
print("have:   "+str(have))

model = RandomForestClassifier()
x_train.drop(["BSHEET10", "BSHEET11", "BSHEET55", "BSHEET56"],inplace=True,axis=1)
x_test.drop(["BSHEET10", "BSHEET11", "BSHEET55", "BSHEET56"],inplace=True,axis=1)
model.fit(x_train,y_train)
result = model.predict(x_test)
nohave = accuracy_score(y_test,result)
print("nohave:    "+str(nohave))
# namelist = Config.name_list
# absense = namelist
# for name in temp["SYMBOL"]:
#     if int(name) in namelist:
#         absense.remove(name)
print("Over")
