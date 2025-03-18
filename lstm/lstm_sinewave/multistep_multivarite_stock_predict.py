"""
Author: Joel
FilePath: lstm/lstm_sinewave/multistep_multivarite_stock_predict.py
Date: 2025-03-13 19:01:29
LastEditTime: 2025-03-15 22:46:00
Description: 
"""
import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# 雅虎财经下载数据
aapl_data = yf.download('AAPL', start='2000-12-12', end='2025-3-14')
aapl_data.to_csv('AAPL.csv')

# 1.加载数据（以苹果股票为例）
data = pd.read_csv('AAPL.csv')  # 从Yahoo Finance下载CSV
# print('aapl_head', data.head())
features = data[['Open', 'High', 'Low', 'Close', 'Volume']].iloc[2:].values
# 为close列单独归一化
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaled = close_scaler.fit_transform(features[:, 3].reshape(-1, 1))
# 数据归一化
feature_scaler = MinMaxScaler()
scaler_features = feature_scaler.fit_transform(features)


# 3.创建时间窗口数据集 多个特征
def create_multivariate_dataset(data, target, time_steps=60):
    X, Y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:i + time_steps, :])  # 包含所有特征
        Y.append(target[i + time_steps])  # 输出仅Close列（已单独归一化）
    return np.array(X), np.array(Y)


time_steps = 60
X, Y = create_multivariate_dataset(scaler_features, close_scaled, time_steps)

print('X.shape', X.shape)
print('Y.shape', Y.shape)
# X = X.reshape(X.shape[0], X.shape[1], 1)

# 4.划分训练集/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# 5.模型构建 输入多个维度
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 6.训练模型
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.1)

# 7.预测测试集并反归一化
predicted = model.predict(X_test)
predicted = close_scaler.inverse_transform(predicted)  # 使用Close列的scaler
# 反归一化真实值
Y_test_actual = close_scaler.inverse_transform(Y_test.reshape(-1, 1))

# 8.可视化结果
plt.figure(figsize=(12, 8))
plt.plot(Y_test_actual, label='True Price')
plt.plot(predicted, label='Predicted Price')
plt.legend()
plt.show()
