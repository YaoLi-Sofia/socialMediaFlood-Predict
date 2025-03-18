"""
Author: Joel
FilePath: lstm/lstm_sinewave/csv_multivariate_stock_predict.py
Date: 2025-03-13 19:01:29
LastEditTime: 2025-03-16 13:49:27
Description: 
"""
import os
from datetime import datetime

import numpy as np
import pandas as pd
from keras import Sequential
from keras.src.layers import LSTM, Dense
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from next_file_name import get_next_predictions_name

# current_datetime = datetime.now()
# print("当前日期和时间：", current_datetime)
# formated_datetime = current_datetime.strftime("%Y-%m-%d")
# print('格式化后的日期和时间：', formated_datetime)

data = yf.download('AAPL', period='10y', interval='1d')
data.to_csv('AAPL.csv')


def load_clean_data():
    # 1. 显式指定日期解析格式
    data = pd.read_csv(
        'AAPL.csv',
        skiprows=2,
        names=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
        index_col='Date',
        parse_dates=True,
        date_parser=lambda x: pd.to_datetime(x, errors='coerce')  # 关键修复
    )

    # 2. 清理无效日期
    data = data[data.index.notnull()]  # 删除日期为NaT的行

    # # 3. 数据类型验证
    # print("日期列类型:", data.index.dtype)  # 应输出 datetime64[ns]

    return data.sort_index()


data = load_clean_data()
# print("数据样例：\n", data.head())

features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
dates = data.index  # 直接获取日期索引 （此时dates是DatetimeIndex类型）
# 为close列单独归一化
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_scaled = close_scaler.fit_transform(features[:, 3].reshape(-1, 1))
# 数据归一化
feature_scaler = MinMaxScaler()
scaler_features = feature_scaler.fit_transform(features)


# 3.创建时间窗口数据集 多个特征
def create_multivariate_dataset(data, target, dates, time_steps=60):
    X, Y, date_list = [], [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  # 包含所有特征
        Y.append(target[i + time_steps])  # 输出仅Close列（已单独归一化）
        date_list.append(dates[i + time_steps])  # 从日期索引中获取
    return np.array(X), np.array(Y), np.array(date_list)


time_steps = 60
X, Y, dates_series = create_multivariate_dataset(scaler_features, close_scaled, dates, time_steps)

# 4. 数据验证
print(f"\n数据维度验证:")
print(f"X形状: {X.shape} | Y形状: {Y.shape} | 日期数量: {len(dates_series)}")

# 4.划分训练集/测试集
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
Y_train, Y_test = Y[:split], Y[split:]
dates_test = dates_series[split:]

# 5.模型构建 输入多个维度
model = Sequential()
model.add(LSTM(100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(100))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# 7. 模型训练
print("\n训练进度：")
history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 7.预测测试集并反归一化
predicted = model.predict(X_test)
predicted = close_scaler.inverse_transform(predicted)  # 使用Close列的scaler
# 反归一化真实值
Y_test_actual = close_scaler.inverse_transform(Y_test.reshape(-1, 1))
# 创建结果表格
results = pd.DataFrame({
    'Date': dates_test,
    'True_Close': Y_test_actual.flatten(),
    'Predicted_Close': predicted.flatten()
})
# # 验证日期类型
# print("结果表中日期类型:", results['Date'].dtype)  # 应输出 datetime64[ns]
# 日期格式化处理
results['Formatted_Date'] = results['Date'].dt.strftime('%Y-%m-%d')

# 9. 结果展示
print("\n预测结果（后5行）：")
print(results[['Formatted_Date', 'True_Close', 'Predicted_Close']].tail(5))

# 保存结果
directory = 'stock_result'
next_name = get_next_predictions_name(base_name='predictions', directory='stock_result', extension='.csv')
results.to_csv(f'{directory}/{next_name}', index=False,
               columns=['Formatted_Date', 'True_Close', 'Predicted_Close'])
# 8.可视化结果
plt.figure(figsize=(14, 7))
plt.plot(results['Date'], results['True_Close'], label='Actual Price', linewidth=2)
plt.plot(results['Date'], results['Predicted_Close'], label='Predicted Price', linestyle='--')
plt.title('Apple Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
