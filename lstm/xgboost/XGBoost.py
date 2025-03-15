import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import gaussian_kde

# 读取数据
file_path = "Train_Data.csv"
df = pd.read_csv(file_path).dropna()

# 划分训练集和测试集（7:3）
train_data, test_data = train_test_split(df, test_size=0.3, random_state=123)

# 提取X和Y
Y_train = train_data.iloc[:, 0].values
X_train = train_data.iloc[:, 1:].values
Y_test = test_data.iloc[:, 0].values
X_test = test_data.iloc[:, 1:].values

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test, label=Y_test)

# 训练 XGBoost GPU 加速模型
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'eta': 0.1,
    'max_depth': 10,
    'device': "cuda"  # GPU加速
}

model_gpu = xgb.train(params, dtrain, num_boost_round=100)

# 预测
Y_train_pred = model_gpu.predict(dtrain)
Y_test_pred = model_gpu.predict(dtest)

# 计算R2和RMSE
train_r2 = r2_score(Y_train, Y_train_pred)
train_rmse = np.sqrt(mean_squared_error(Y_train, Y_train_pred))
test_r2 = r2_score(Y_test, Y_test_pred)
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))

print(f"训练集 R2: {train_r2:.4f}, RMSE: {train_rmse:.4f}")
print(f"测试集 R2: {test_r2:.4f}, RMSE: {test_rmse:.4f}")

# 绘制训练集密度散点图
# 计算训练集点密度
xy = np.vstack([Y_train, Y_train_pred])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = Y_train[idx], Y_train_pred[idx], z[idx]
fig, ax = plt.subplots()
plt.scatter(Y_train, Y_train_pred, c=z, cmap='Spectral_r')
Train_min_value = min(min(Y_train), min(Y_train_pred))
Train_max_value = max(max(Y_train), max(Y_train_pred))
plt.plot([Train_min_value, Train_max_value], [Train_min_value, Train_max_value],
         color='red', linestyle='--', linewidth=2)
plt.colorbar(label="Density")
plt.suptitle('Training Set', y=1.02)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.axis("square")
plt.show()

# 绘制测试集密度散点图
# 计算测试集点密度
xy = np.vstack([Y_test, Y_test_pred])
z = gaussian_kde(xy)(xy)
idx = z.argsort()
x, y, z = Y_test[idx], Y_test_pred[idx], z[idx]
fig, ax = plt.subplots()
plt.scatter(Y_test, Y_test_pred, c=z, cmap='Spectral_r')
Test_min_value = min(min(Y_test), min(Y_test_pred))
Test_max_value = max(max(Y_test), max(Y_test_pred))
plt.plot([Test_min_value, Test_max_value], [Test_min_value, Test_max_value],
         color='red', linestyle='--', linewidth=2)
plt.colorbar(label="Density")
plt.suptitle('Test Set', y=1.02)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.axis("square")
plt.show()

# 计算 SHAP 值
explainer = shap.TreeExplainer(model=model_gpu)
shap_values = explainer.shap_values(X_train)

# 绘制 SHAP Summary Plot
shap.summary_plot(shap_values, X_train)

# 绘制 SHAP 重要性图
shap.summary_plot(shap_values, X_train, plot_type="bar")

# 绘制 部分依赖 (PDP) 图
shap.dependence_plot('Feature 18', shap_values, X_train, interaction_index=None)
