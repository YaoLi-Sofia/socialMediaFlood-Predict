"""
Author: Joel
FilePath: lstm/weather/latest_precip_predict.py
Date: 2025-03-15 18:15:21
LastEditTime: 2025-03-18 15:35:49
Description: 
"""
from datetime import datetime, timedelta

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import next_file_name

# 读取文件
weather_data = pd.read_csv(
    '50353_update.csv',
    sep=';',
    # names=
    # list(pd.read_csv('50353_update.csv').columns.values.tolist()[0]
    #            .replace(';', ',').strip(',').split(',')),
    # ["Date", "Precip", "Wsmean", "Tmean", "Rhumean", "Rhumin", "Sunhour", "Tmin", "Tmax"],
    parse_dates=True,
    date_parser=lambda x: pd.to_datetime(x, errors='coerce')
)

weather_data = weather_data[weather_data['Date'].notnull()]
# weather_data = weather_data.sort_values(by='Date', ascending=True)
# print('weather_data', weather_data)
# 加载数据
# features = weather_data.iloc[:, 1:].values
# 选择所有数值型特征列（排除Site）
# print("数据列名:", weather_data.columns.tolist())

features = weather_data[
    ['Precip', 'Wsmean', 'Tmean', 'Rhumean', 'Rhumin',
     'Sunhour', 'Tmin', 'Tmax']
].values

# print('features', features)

# 预处理-数据归一化
feature_scaler = MinMaxScaler()
scaler_features = feature_scaler.fit_transform(features)

# 单独归一化precip
precip_scaler = MinMaxScaler(feature_range=(0, 1))
precip_values = weather_data['Precip'].values.reshape(-1, 1)
scaler_precip = precip_scaler.fit_transform(precip_values)

dates = weather_data['Date'].values.tolist()


# 创建多个特征的时间窗口数据集
def create_multivariate_dataset(data, target, dates, time_steps=60):
    X, Y, date_list = [], [], []
    for i in range(len(data) - time_steps - 1):
        X.append(data[i:i + time_steps, :])
        Y.append(target[i + time_steps])
        date_list.append(dates[i + time_steps])
    return np.array(X), np.array(Y), np.array(date_list)


time_steps = 60
X, Y, dates_series = create_multivariate_dataset(scaler_features, scaler_precip, dates, time_steps)

print(f"验证数据集：\nX.shape: {X.shape}, Y.shape: {Y.shape}, 日期: {len(dates_series)}")

# 时序划分训练集和测试集
spilt = int(len(X) * 0.8)
X_train, X_test = X[:spilt], X[spilt:]
Y_train, Y_test = Y[:spilt], Y[spilt:]
dates_test = dates_series[spilt:]

# 构建模型
model = Sequential()
model.add(LSTM(units=150, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(LSTM(units=150))
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

results = pd.DataFrame({
    'Date': dates_test,
    'Actually_Precip': Y_test_actual.flatten(),
    'Predicted_Precip': predicted.flatten(),
})

print(f"results['Date']的类型1：{results['Date'].dtype}")
results['Date'] = pd.to_datetime(results['Date'])
print(f"results['Date']的类型2：{results['Date'].dtype}")
results['Formatted_Date'] = (results['Date'].dt.strftime('%Y-%m-%d'))

predict_steps = 5
latest_data = scaler_features[-time_steps:]
future_predictions = []
current_window = latest_data.copy()

for _ in range(predict_steps):
    pred = model.predict(current_window.reshape(1, time_steps, -1))
    future_predictions.append(pred[0, 0])
    new_row = current_window[-1].copy()
    new_row[1] = pred[0, 0]
    current_window = np.vstack((current_window[1:], new_row))
future_predictions = precip_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
# 将日期字符串转换为 datetime 对象
dates_test = [datetime.strptime(date_str, '%Y/%m/%d') for date_str in dates_test]
print('dates_test[-5]', dates_test[-5])
future_dates = [dates_test[-1] + timedelta(days=i + 1) for i in range(predict_steps)]

future_result = pd.DataFrame({
    'Date': future_dates,
    'Actually_Precip': [np.nan] * predict_steps,
    'Predicted_Precip': future_predictions.flatten(),
})

finally_result = pd.concat([results, future_result], axis=0)

finally_result['Date'] = pd.to_datetime(finally_result['Date'])
finally_result['Formatted_Date'] = finally_result['Date'].dt.strftime('%Y-%m-%d')

print(
    f"预览结果显示(倒数8条)：\n {finally_result[['Formatted_Date', 'Actually_Precip', 'Predicted_Precip']].tail(predict_steps + 3)}")

directory = 'results'
base_name = 'predicted_precip'
next_file_name = next_file_name.get_next_file_name(directory='results', base_name='predicted_precip', extension='.csv')
finally_result.to_csv(f"{directory}/{next_file_name}", index=False,
                      columns=['Formatted_Date', 'Actually_Precip', 'Predicted_Precip'])

print(f"最终结果已保存，路径：{directory}/{base_name}")

# 可视化
plt.figure(figsize=(14, 8))
plt.plot(
    finally_result[finally_result['Actually_Precip'].notna()]['Date'],
    finally_result[finally_result['Actually_Precip'].notna()]['Actually_Precip'],
    label='Actually_Precip (History)',
    color='blue',
    linewidth=2
)
plt.plot(
    finally_result[finally_result['Actually_Precip'].notna()]['Date'],
    finally_result[finally_result['Actually_Precip'].notna()]['Predicted_Precip'],
    label='Predicted_Precip (History)',
    color='orange',
    linestyle='--'
)
plt.plot(
    finally_result[finally_result['Actually_Precip'].isna()]['Date'],
    finally_result[finally_result['Actually_Precip'].isna()]['Predicted_Precip'],
    label='Predicted_Precip (Future)',
    color='red',
    linestyle='--',
    marker='o'
)
plt.title('Precip Prediction', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Precip (mm)', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
