# Taxi-Prediction
Predict the taxi's parking place(Pattern Recognition's homework)

## raw_data
用于保存原始出租车轨迹数据

## data
用于保存处理之后的数据，用于模型的训练和验证

## models
用于保存训练过的模型.hdf5

## submissions
用于保存最终生成的文件或者图表

## logs
用于保存训练过程中模型的准确率变化情况

## util
数据清洗预处理的代码，将raw_data中的数据处理之后存放到data

## .*
模型训练代码
cab_embedding.py embedding特征之后加入到多层神经网络中进行训练
cab_mlp.py 单层神经网络训练
cab_rnn.py 直接用循环神经网络进行训练
