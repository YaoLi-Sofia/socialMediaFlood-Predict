"""
Author: Joel
FilePath: nature/lstm/lstm_sinewave/sine_wave_predict_update.py
Date: 2025-03-13 18:30:24
LastEditTime: 2025-03-14 17:43:42
Description: 
"""
import numpy as np
import matplotlib.pyplot as plt
from keras.src.layers import TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, LSTM, RepeatVector, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 生成正弦波数据
x = np.linspace(0, 100, 1000)  # 生成0到100之间的1000个等距点;
y = np.sin(x)  # 计算每个点的正弦值;


# 数据预处理
# 将数据转换为监督学习格式
def create_dataset(data, time_step=10, predict_steps=5, stride=1):
    X, Y = [], []
    max_i = len(data) - time_step - predict_steps + 1
    if max_i < 0:
        raise ValueError("数据长度不足！需至少 {} 点，当前 {} 点".format(
            time_step + predict_steps, len(data)))

    for i in range(0, max_i, stride):
        X.append(data[i:i + time_step])  # 取连续10个点作为输入
        Y.append(data[i + time_step:i + time_step + predict_steps])
    return np.array(X), np.array(Y)


time_step = 10
predict_steps = 100

X, Y = create_dataset(y, time_step, predict_steps, stride=1)
if len(X) == 0:
    raise ValueError('数据处理异常，请检查代码！')
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)

X = X.reshape(X.shape[0], X.shape[1], 1)
Y = Y.reshape(Y.shape[0], Y.shape[1], 1)

# 构建模型（Seq2Seq结构）
model = Sequential()
model.add(LSTM(256, input_shape=(time_step, 1)))
# model.add(Dropout(0.2))
model.add(RepeatVector(predict_steps))
model.add(LSTM(256, return_sequences=True))
model.add(LSTM(128, return_sequences=True))
# model.add(TimeDistributed(Dense(1, activation='relu')))
model.add(TimeDistributed(Dense(1)))
# model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# 按时间顺序划分数据集
split_index = int(len(X) * 0.8)
X_train, X_val = X[:split_index], X[split_index:]
Y_train, Y_val = Y[:split_index], Y[split_index:]

# 监控损失
callbacks = [
    EarlyStopping(monitor='val_loss', patience=8, verbose=1, mode='min'),
    ModelCheckpoint('best_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'),
]

# 训练模型 使其学会从输入序列预测下一个点
history = model.fit(
    X_train,
    Y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, Y_val),
    callbacks=callbacks,
    verbose=1,
)
# model.fit(
#     X,
#     Y,
#     epochs=50,
#     batch_size=32,
#     verbose=1,
# )

# 预测并可视化
test_input = y[-time_step:].reshape(1, time_step, 1)  # 取最后10个点作为初始输入
# print('test_input.shape:', test_input.shape) # 预期[1,10,1]
prediction = model.predict(test_input)[0, :, 0]

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(y, label='Real')
plt.plot(range(len(y), len(y) + predict_steps), prediction, label='Predicted')
plt.legend()
plt.show()
