from Project.src.config import Config
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def model_training():
    name_list = Config.name_list
    for name in name_list:
        data = pd.read_csv("../data/processed_data/" + str(name) + "__processed_data.csv")
        train_date_index = [num for num in range(1, 90)]
        train = data.iloc[:, train_date_index]
        # target = data.iloc[:,91]
        target = data.iloc[:, [num for num in range(91, data.shape[1])]]
        score = 0
        for num in range(6):
            labelname = ["p5", "p10", "p15", "p-5", "p-10", "p-15"]
            x_train, x_test, y_train, y_test = train_test_split(train, target.iloc[:, num], test_size=0.3)
            clf = RandomForestClassifier()
            clf.fit(x_train, y_train)
            result = clf.predict(x_test)
            pre_result = clf.predict_proba(x_test)
            if pre_result.shape[1]==2:
                out_put = pd.DataFrame(
                    data={labelname[num]: y_test, "predict": result, "predict_proba_False": pre_result[:, 0],
                          "predict_proba_True": pre_result[:, 1]})
            else:
                if result[0]:
                    out_put = pd.DataFrame(
                        data={labelname[num]: y_test, "predict": result, "predict_proba_False": [0 for num in range(pre_result.shape[0])],
                              "predict_proba_True": pre_result[:, 0]})
                else:
                    out_put = pd.DataFrame(
                        data={labelname[num]: y_test, "predict": result, "predict_proba_False": pre_result[:, 0],
                              "predict_proba_True": [0 for num in range(pre_result.shape[0])]})

            out_put.to_csv("../data/result/" + str(name) + "__" + str(labelname[num]) + "__" + "_processed_data.csv")
            with open("accuracy.txt", "a+") as f:
                f.write(str(name) + " fund " + labelname[num] + " RF_accuracy:" + str(
                    accuracy_score(y_test, result)) + " \n")
        #print(score / 6)
        print(name)


model_training()
