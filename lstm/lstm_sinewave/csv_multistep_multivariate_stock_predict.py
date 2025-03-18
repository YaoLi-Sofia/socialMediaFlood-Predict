"""
Author: Joel
FilePath: lstm/lstm_sinewave/csv_multistep_multivariate_stock_predict.py
Date: 2025-03-13 19:01:29
LastEditTime: 2025-03-17 22:01:38
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
from pandas.tseries.offsets import BDay

data = yf.download('AAPL', period='10y', interval='1d')
data.to_csv('AAPL.csv')


def load_clean_data():
    # 1. 显式指定日期解析格式
    data = pd.read_csv(
        'AAPL.csv',
        skiprows=2,
        names=['Date'] + pd.read_csv('AAPL.csv').columns.values.tolist()[1:],
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
# print("数据样例：\n", data.tail(10))

features = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
dates = data.index  # 直接获取日期索引 （此时dates是DatetimeIndex类型）
# print('dates', dates)
# 为close列单独归一化
close_scaler = MinMaxScaler(feature_range=(0, 1))
close_values = data['Close'].values.reshape(-1, 1)
scaler_close = close_scaler.fit_transform(close_values)
# 数据归一化
feature_scaler = MinMaxScaler()
scaler_features = feature_scaler.fit_transform(features)


# 3.创建时间窗口数据集 多个特征
def create_multivariate_dataset(data, target, dates, time_steps=60, ):
    X, Y, date_list = [], [], []
    # print('len(data)', len(data)) 2516
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps, :])  # 包含所有特征
        Y.append(target[i + time_steps])  # 输出仅Close列（已单独归一化）
        date_list.append(dates[i + time_steps])  # 从日期索引中获取
    return np.array(X), np.array(Y), np.array(date_list)


time_steps = 60
predict_steps = 5
X, Y, dates_series = create_multivariate_dataset(scaler_features, scaler_close, dates, time_steps)

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
    epochs=2,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 7.预测测试集
predicted = model.predict(X_test)
# 反归一化所有预测步
predicted = close_scaler.inverse_transform(predicted)  # 使用Close列的scaler
# 反归一化真实值
Y_test_actual = close_scaler.inverse_transform(Y_test.reshape(-1, 1))

# 4. 数据验证
print(f"\n数据维度验证:")
print(f"predicted形状: {predicted.shape} | Y_test_actual形状: {Y_test_actual.shape} | 日期数量: {len(dates_test)}")

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

# 获取最新60天的归一化数据
latest_data = scaler_features[-time_steps:]
# 递归预测未来
future_predictions = []
current_window = latest_data.copy()  # 初始形状 (60, 5)

for _ in range(predict_steps):
    pred = model.predict(current_window.reshape(1, time_steps, -1))  # 输出形状 (1, 5)
    # print('pred', pred)
    # print('pred.shape', pred.shape)
    future_predictions.append(pred[0, 0])
    # print('future_predictions', future_predictions)
    # 生成新行：用预测的Close更新最后一行
    new_row = current_window[-1].copy()  # 复制最后一行特征
    print('new_row', new_row)
    new_row[3] = pred[0][0]  # 更新Close列
    # print('new_row[0]', new_row[0])
    # print('new_row[1]', new_row[1])
    # print('new_row[2]', new_row[2])
    # print('new_row[3]', new_row[3])
    # print('new_row[4]', new_row[4])
    current_window = np.vstack((current_window[1:], new_row))
    # print('current_window', current_window)
# 反归一化
future_predictions = close_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

last_date = data.index[-1]
# print('last_date', last_date)
# for y in range(predict_steps):
#     abc = last_date + BDay(y + 1)
#     print('abc', abc)
future_dates = [last_date + BDay(i + 1) for i in range(predict_steps)]

future_results = pd.DataFrame({
    'Date': future_dates,
    'True_Close': [np.nan] * predict_steps,  # 无真实值
    'Predicted_Close': future_predictions.flatten(),
    'Formatted_Date': [date.strftime('%Y-%m-%d') for date in future_dates]
})

final_results = pd.concat([results, future_results], axis=0)  # axis=0 按行合并 1 按列合并

# 9. 结果展示
print("\n预测结果（后8行）：")
print(final_results[['Formatted_Date', 'True_Close', 'Predicted_Close', ]].tail(8))

# 保存结果
directory = 'stock_result'
next_name = get_next_predictions_name(base_name='predictions', directory='stock_result', extension='.csv')
final_results.to_csv(f'{directory}/{next_name}', index=False,
                     columns=['Formatted_Date', 'True_Close', 'Predicted_Close', ])
print(f"全部结果已经保存，路径：{directory}/{next_name}")
# 8.可视化结果
plt.figure(figsize=(14, 7))
# 绘制历史真实值
plt.plot(
    final_results[final_results['True_Close'].notna()]['Date'],
    final_results[final_results['True_Close'].notna()]['True_Close'],
    label='True Close (History)',
    color='blue',
    linewidth=2
)

# 绘制历史预测值
plt.plot(
    final_results[final_results['True_Close'].notna()]['Date'],
    final_results[final_results['True_Close'].notna()]['Predicted_Close'],
    label='Predicted Close (History)',
    linestyle='--',
    color='orange'
)

# 绘制未来预测值
plt.plot(
    final_results[final_results['True_Close'].isna()]['Date'],
    final_results[final_results['True_Close'].isna()]['Predicted_Close'],
    label='Predicted Close (Future)',
    linestyle='--',
    color='red',
    marker='o'
)
plt.title('Apple Stock Price Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
