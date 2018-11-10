from src.config import Config
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def model_training():
    name_list = Config.name_list
    score_dict = {"p5": [], "p10": [], "p15": [], "p-5": [], "p-10": [], "p-15": [], "num": []}
    with open("accuracy.txt", "w") as f:
        f.seek(0)
        f.truncate()
    for name in name_list:
        data = pd.read_csv("../data/processed_data/" + str(name) + "__processed_data.csv")
        data = data[(data["train"] >= 45) & (data["test"] >= 15)]
        train = data.iloc[:, 1:90]
        target = data.iloc[:, -6:]
        score_dict["num"].append(len(data))
        for num in range(6):
            labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
            x_train, x_test, y_train, y_test = train_test_split(train, target.iloc[:, num], test_size=0.3)
            for index in range(4):
                temp = [x_train, x_test, y_train, y_test]
                temp_name = ["x_train", "x_test", "y_train", "y_test"]
                temp[index].to_csv(
                    "../data/temp/" + str(name) + "__" + str(labelname[num]) + "__" + temp_name[index] + "__data.csv",
                    index=False)
            # x_train.to_csv("../data/temp/" + str(name) + "__" + str(labelname[num]) + "__" + "_x-train_data.csv",
            #                index=False)
            # x_test.to_csv("../data/temp/" + str(name) + "__" + str(labelname[num]) + "__" + "_x-test_data.csv",
            #               index=False)
            # y_train.to_csv("../data/temp/" + str(name) + "__" + str(labelname[num]) + "__" + "_y-train_data.csv",
            #                index=False)
            # y_test.to_csv("../data/temp/" + str(name) + "__" + str(labelname[num]) + "__" + "_y-test_data.csv",
            #               index=False)
            clf = RandomForestClassifier()
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            pre_result = clf.predict_proba(x_test)
            score = accuracy_score(y_test, result)
            score_dict[labelname[num]].append(score)
            if pre_result.shape[1] == 2:
                proba = [[temp[0], temp[1]] for temp in pre_result]
            else:
                if result[0]:
                    proba = [[0, 1] for temp in range(len(pre_result))]
                else:
                    proba = [[1, 0] for temp in range(len(pre_result))]
            out_put = pd.DataFrame({"result": result, "proba": proba})
            # if pre_result.shape[1] == 2:
            #     out_put = pd.DataFrame(
            #         data={labelname[num]: y_test, "predict": result, "predict_proba_False": pre_result[:, 0],
            #               "predict_proba_True": pre_result[:, 1]})
            # else:
            #     if result[0]:
            #         out_put = pd.DataFrame(
            #             data={labelname[num]: y_test, "predict": result,
            #                   "predict_proba_False": [0 for num in range(pre_result.shape[0])],
            #                   "predict_proba_True": pre_result[:, 0]})
            #     else:
            #         out_put = pd.DataFrame(
            #             data={labelname[num]: y_test, "predict": result, "predict_proba_False": pre_result[:, 0],
            #                   "predict_proba_True": [0 for num in range(pre_result.shape[0])]})
            out_put.to_csv("../data/result/" + str(name) + "__" + str(labelname[num]) + "__" + "_processed_data.csv",
                           index=False)
            with open("accuracy.txt", "a+") as f:
                f.write(str(name) + " fund " + labelname[num] + " RF_accuracy:" + str(score) + " \n")
        print(name)
    pd.DataFrame(data=score_dict).to_csv("score_dict.csv")


#model_training()



# labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
# for label in labelname:
#     for type in ["y_train", "y_test"]:
#         temp = []
#         for name in Config.name_list:
#             data = pd.read_csv("../data/temp/" + str(name) + "__" + str(label) + "__" + type + "__data.csv",names=["target"])
#             temp.append(data)
#         result = temp[0]
#         for num in range(1, len(temp)):
#             result = result.append(temp[num])
#         result.to_csv("__" + str(label) + "__" + type + "__data.csv", index=False)
#         print("__" + str(label) + "__" + type + "__data.csv   is  Over")

labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
score_dic = {"p5": [], "p10": [], "p15": [], "p-5": [], "p-10": [], "p-15": []}
for label in labelname:
    x_train = pd.read_csv("__" + str(label) + "__" + "x_train" + "__data.csv")
    y_train = pd.read_csv("__" + str(label) + "__" + "y_train" + "__data.csv")
    x_test = pd.read_csv("__" + str(label) + "__" + "x_test" + "__data.csv")
    y_test = pd.read_csv("__" + str(label) + "__" + "y_test" + "__data.csv")
    model = RandomForestClassifier()
    model.fit(x_train,y_train)
    result = model.predict(x_test)
    #score = accuracy_score(y_test,result)
    score_dic[label].append(accuracy_score(y_test,result))
#pd.DataFrame().to_csv("all__score.csv", index=False)
output = pd.DataFrame(score_dic)
output.to_csv("all_acore.csv")
#print("__" + str(label) + "__" + type + "__data.csv   is  Over")

