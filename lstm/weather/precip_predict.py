"""
Author: Joel
FilePath: lstm/weather/precip_predict.py
Date: 2025-03-15 18:15:21
LastEditTime: 2025-03-15 20:13:19
Description: 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 读取文件
weather_data = pd.read_csv('50353_update.csv', sep=';')

# 加载数据
# features = weather_data.iloc[:, 1:].values
# 选择所有数值型特征列（排除Site）
# print("数据列名:", weather_data.columns.tolist())

features = weather_data[
    ['Year', 'Month', 'Day', 'Precip', 'Wsmean', 'Tmean', 'Rhumean', 'Rhumin',
     'Sunhour', 'Tmin', 'Tmax']
].values

# print('features.shape', features.shape)

# 预处理-数据归一化
feature_scaler = MinMaxScaler()
scaler_features = feature_scaler.fit_transform(features)

# 单独归一化precip
precip_scaler = MinMaxScaler(feature_range=(0, 1))
scaler_precip = precip_scaler.fit_transform(features[:, 4].reshape(-1, 1))


# 创建多个特征的时间窗口数据集
def create_multivariate_dataset(data, target, time_steps=60):
    X, Y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:i + time_steps, :])
        Y.append(target[i + time_steps])
    return np.array(X), np.array(Y)


time_steps = 60
X, Y = create_multivariate_dataset(scaler_features, scaler_precip, time_steps)

# 时序划分训练集和测试集
spilt = int(len(X) * 0.8)
X_train, X_test = X[:spilt], X[spilt:]
Y_train, Y_test = Y[:spilt], Y[spilt:]

# 构建模型
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=100))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 训练模型
model.fit(
    X_train,
    Y_train,
    epochs=100,
    verbose=1,
    batch_size=32,
    validation_split=0.1
)
# 预测
predicted = model.predict(X_test)
# 反归一化
predicted = precip_scaler.inverse_transform(predicted)
# 反归一化真实值
Y_test_actual = precip_scaler.inverse_transform(Y_test.reshape(-1, 1))

# 可视化
plt.plot(predicted, label='predicted precip')
plt.plot(Y_test_actual, label='actually precip')
plt.legend()
plt.show()
