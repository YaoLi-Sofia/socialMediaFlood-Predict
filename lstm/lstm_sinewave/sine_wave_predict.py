"""
Author: Joel
FilePath: nature/lstm/lstm_sinewave/sine_wave_predict.py
Date: 2025-03-13 18:30:24
LastEditTime: 2025-03-14 13:09:44
Description: 
"""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import Sequential

# NumPy（Numerical Python）是Python的数值计算库，提供多维数组对象（ndarray）和数学函数;
# matplotlib.pyplot  Python的绘图库，用于数据可视化;
# Keras是深度学习框架，简化了神经网络构建流程，集成在TensorFlow中;
# Sequential：线性堆叠层的模型容器;
# LSTM：长短期记忆网络层，用于处理时序数据;
# Dense：全连接层，用于输出预测值;

# 1、生成正弦波数据
x = np.linspace(0, 100, 1000)  # 生成0到100之间的1000个等距点;
y = np.sin(x)  # 计算每个点的正弦值;


# np.linspace()
# 功能：在指定区间生成等距数值;
# 参数：
# start：起始值（0）;
# stop：结束值（100）;
# num：生成样本数（1000）;
# 输出：形如 [0.0, 0.1, 0.2, ..., 99.9, 100.0] 的数组;

# np.sin()
# 功能：计算数组中每个元素的正弦值;
# 输出：形如 [sin(0), sin(0.1), ..., sin(100)] 的正弦波数据;

# 2、数据预处理
# 2.1、将数据转换为监督学习格式（用前10个时间步骤预测下一步）
def create_dataset(data, time_step=10):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:i + time_step])  # 取连续10个点作为输入
        Y.append(data[i + time_step])  # 第11个点作为输出
    return np.array(X), np.array(Y)


# 监督学习格式:
# 目的：将时序数据转换为“输入-输出”对，使模型学习用过去预测未来;
# 示例：
# 原始数据：[1, 2, 3, 4, 5]，time_step=2
# 输入 X：[[1,2], [2,3], [3,4]]
# 输出 Y：[3, 4, 5]
# range(len(data) - time_step)
# 作用：确保最后一个输入序列不超出数据范围;

time_step = 10
X, Y = create_dataset(y, time_step)

print("X.shape:", X.shape)  # 应输出类似 (990, 10, 1)
print("Y.shape:", Y.shape)  # 应输出类似 (990,)

# 调整输入形状为 [样本数，时间步长，特征数]
# 样本数：X.shape[0]（即总共有多少个输入样本）
# 时间步长：X.shape[1]（即 time_steps=10）
# 特征数：1（每个时间点只有1个特征，即正弦波的值）

X = X.reshape(X.shape[0], X.shape[1], 1)

# reshape()
# 功能：改变数组形状，不改变数据内容;
# 参数：
# X.shape[0]：样本数（如990）;
# X.shape[1]：时间步长（10）;
# 1：特征数（每个时间点的数值）;

# 构建LSTM模型
# LSTM层：
# units=50：LSTM层有50个神经元（可以理解为模型的复杂度）
# activation='relu'：使用ReLU激活函数（比默认的tanh更简单，适合本任务）
# input_shape=(time_steps, 1)：输入形状为 (10, 1)。
# Dense层：
# Dense(1)：输出层，1个神经元（预测一个值）
# 编译：
# optimizer='adam'：使用Adam优化器（自适应学习率，适合大多数任务）
# loss='mse'：均方误差（Mean Squared Error），用于回归问题

model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(time_step, 1)))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Sequential模型
# 作用：按顺序逐层堆叠神经网络层;
# 类比：像搭积木，逐层添加组件;
# LSTM层
# 参数：
# units=50：LSTM单元（神经元）数量，控制模型复杂度;
# activation='relu'：激活函数，引入非线性（ReLU比默认的tanh更简单高效）;
# input_shape=(time_step, 1)：输入数据形状（无需指定样本数）;
# Dense层
# 作用：全连接层，将LSTM输出映射到预测值;
# units=1：输出1个数值（预测的下一个时间点值）;
# compile()
# 参数：
# loss='mse'：均方误差（Mean Squared Error），衡量预测值与真实值的差距;
# optimizer='adam'：自适应学习率优化器，适合大多数任务;


# print("X.shape:", X.shape)  # 应输出类似 (990, 10, 1)
# print("Y.shape:", Y.shape)  # 应输出类似 (990,)

# 训练模型 使其学会从输入序列预测下一个点
# 训练过程中，模型会不断调整参数，最小化预测值和真实值的均方误差（MSE）
model.fit(X, Y, epochs=20, verbose=1)

# fit()
# 参数：
# X：输入数据（形状 (样本数, 10, 1)）;
# Y：标签数据（形状 (样本数,)）;
# epochs=20：整个数据集遍历20次;
# verbose=1：显示训练进度条;

# 预测并可视化
test_input = y[-time_step:].reshape(1, time_step, 1)  # 取最后10个点作为初始输入
# print('test_input.shape:', test_input.shape) # 预期[1,10,1]
prediction = []
for _ in range(50):  # 预测未来50步
    pred = model.predict(test_input)  # 预测下一步
    prediction.append(pred[0, 0])  # 记录预测值
    pred = pred.reshape(1, 1, 1)  # 确保维度匹配
    # 更新输入：去掉最旧的点，加入最新预测值
    test_input = np.concatenate([test_input[:, 1:, :], pred], axis=1)

# model.predict()
# 功能：用当前输入预测下一个点;
# 输出：形状为 (1,1)（1个样本，1个输出值）;
# np.concatenate()
# 作用：沿时间步维度（axis=1）拼接数据;
# 示例：
# 原输入：[[t1, t2, ..., t10]]（形状 (1,10,1)）;
# 预测值：t11 → 新输入变为 [[t2, t3, ..., t11]];

# 可视化
plt.plot(y, label='Real')
plt.plot(range(len(y), len(y) + 50), prediction, label='Predicted')
plt.legend()
plt.show()

# plt.plot()
# 功能：绘制曲线;
# 参数：
# y：真实正弦波数据;
# prediction：预测的50个点;
# plt.legend()
# 作用：显示图例（label 参数内容）;
# plt.show()
# 作用：显示图像窗口;
