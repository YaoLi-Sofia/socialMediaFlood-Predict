"""
Author: Joel
FilePath: nature/lstm/lstm_sinewave/stock_predict.py
Date: 2025-03-13 19:01:29
LastEditTime: 2025-03-14 21:47:02
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
prices = data['Close'].iloc[2:].values.reshape(-1, 1)

# 2.数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)


# 3.创建时间窗口数据集
def create_dataset(data, time_steps=60):
    X, Y = [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:(i + time_steps), 0])
        Y.append(data[i + time_steps, 0])
    return np.array(X), np.array(Y)


time_steps = 60
X, Y = create_dataset(scaled_prices, time_steps)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 4.划分训练集/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]

# 5.构建更复杂的LSTM模型
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_steps, 1)))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 6.训练模型
model.fit(X_train, Y_train, epochs=50, batch_size=32, validation_split=0.1)

# 7.预测测试集
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

# 8.可视化结果
plt.plot(Y_test_actual, label='True Price')
plt.plot(predicted, label='Predicted Price')
plt.legend()
plt.show()
