import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 加载波士顿房价数据集
boston = pd.read_csv('D:/boston_house_prices.csv')

# 提取 "RM" 作为特征
X = boston.iloc[:, 5].values.reshape(-1, 1)  # 使用 "RM" (房间数量)
y = boston.iloc[:, -1].values  # 目标变量

# 添加偏置项
X_b = np.c_[np.ones((X.shape[0], 1)), X]  # 在特征前添加一列 1

# 最小二乘法计算参数
theta_ls = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# 预测
y_predict_ls = X_b.dot(theta_ls)

# 梯度下降参数
learning_rate = 0.01
n_iterations = 1000
m = len(y)

# 初始化参数
theta_gd = np.random.randn(2, 1)  # 两个参数，包含偏置项

# 梯度下降
for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta_gd) - y.reshape(-1, 1))
    theta_gd -= learning_rate * gradients

# 预测
y_predict_gd = X_b.dot(theta_gd)

# 绘制结果
plt.scatter(X, y, color='blue', label='Data points')  # 原始数据点
plt.plot(X, y_predict_ls, color='red', label='Least Squares Fit')  # 最小二乘法拟合线
plt.plot(X, y_predict_gd, color='green', label='Gradient Descent Fit')  # 梯度下降拟合线
plt.xlabel('Number of Rooms (RM)')
plt.ylabel('House Price')
plt.title('Linear Regression: Least Squares vs Gradient Descent')
plt.legend()
plt.show()
