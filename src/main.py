import data_process_nav
import nav_test_data_process
import model_training_prediction
import fund_type_classfier
import data_process_manger

if __name__ == '__main__':
    # 对原始的净值数据进行处理
    data_process_nav.data_process()
    # 添加基金经理特征
    data_process_manger.data_process()
    # 把所有数据划分成股票基金数据和债券基金数据
    fund_type_classfier.data_process()
    # # 对2018年的数据进行处理
    # nav_test_data_process.data_process()
    # 划分测试集和训练集
    model_training_prediction.train_test_split()
    # 训练模型并保存模型
    model_training_prediction.model_training()
    # 利用已保存的数据进行训练并输出准确率
    model_training_prediction.model_prediction()
