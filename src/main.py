import nav_train_data_process
import nav_test_data_process
import model_training_prediction
import fund_type_classfier

if __name__ == '__main__':
    # 对原始的净值数据进行处理
    nav_train_data_process.data_process()
    # 把所有数据划分成股票基金数据和债券基金数据
    fund_type_classfier.data_process()
    # 对2018年的数据进行处理
    nav_test_data_process.data_process()
    # 训练模型并保存模型
    model_training_prediction.model_training()
    # 利用已保存的数据进行训练并输出准确率
    model_training_prediction.model_prediction()
