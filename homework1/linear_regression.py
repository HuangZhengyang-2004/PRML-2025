import numpy as np
import matplotlib.pyplot as plt
import os

# 数据加载
train_data = np.loadtxt(r'E:\TESTpy\PRML\homework1\data\train.txt',
                        skiprows=1,  # 跳过标题行
                        delimiter='\t')  # 指定制表符分隔
test_data = np.loadtxt(r'E:\TESTpy\PRML\homework1\data\test.txt',
                       skiprows=1,
                       delimiter='\t')
X_train, y_train = train_data[:, 0], train_data[:, 1]
X_test, y_test = test_data[:, 0], test_data[:, 1]

# 添加偏置项
X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
X_test_b = np.c_[np.ones((len(X_test), 1)), X_test]

# 最小二乘法
def least_squares(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# 梯度下降法
def gradient_descent(X, y, lr=0.01, n_iter=1000):
    w = np.random.randn(X.shape[1])
    losses = []
    for _ in range(n_iter):
        y_pred = X @ w
        gradient = (2 / X.shape[0]) * X.T @ (y_pred - y)
        w -= lr * gradient
        losses.append(np.mean((y_pred - y) ** 2))
    return w, losses

# 牛顿法
def newton_method(X, y, n_iter=10):
    w = np.random.randn(X.shape[1])
    H = 2 / X.shape[0] * X.T @ X  # Hessian矩阵
    for _ in range(n_iter):
        grad = 2 / X.shape[0] * X.T @ (X @ w - y)
        w -= np.linalg.inv(H) @ grad
    return w

# 模型训练和评估
methods = {
    "Least Squares": least_squares,
    "Gradient Descent": gradient_descent,
    "Newton Method": newton_method
}

# 在绘图前创建结果目录
results_dir = r'E:\TESTpy\PRML\homework1\results'
os.makedirs(results_dir, exist_ok=True)

# 初始化对比图
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, s=10, label='Train Data', zorder=3)

for name, method in methods.items():
    # 训练模型
    if name == "Gradient Descent":
        w, losses = method(X_train_b, y_train)
        # 保存GD的损失曲线
        plt.figure()
        plt.plot(losses)
        plt.title('Gradient Descent Loss Curve')
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.savefig(os.path.join(results_dir, 'gd_loss_curve.png'))
        plt.close()
    else:
        w = method(X_train_b, y_train)

    # 预测和评估
    train_pred = X_train_b @ w
    test_pred = X_test_b @ w
    train_mse = np.mean((train_pred - y_train) ** 2)
    test_mse = np.mean((test_pred - y_test) ** 2)

    # 为每个方法单独保存拟合图
    plt.figure(figsize=(8, 5))
    plt.scatter(X_train, y_train, s=10, label='Train Data')
    plt.plot(X_train, train_pred, 'r-', linewidth=2, label=f'{name} Fit')
    plt.title(f'{name}\nTrain MSE: {train_mse:.2f}, Test MSE: {test_mse:.2f}')
    plt.legend()
    plt.savefig(os.path.join(results_dir, f'linear_fit_{name.replace(" ", "_")}.png'))
    plt.close()

    # 添加到对比图
    plt.plot(X_train, train_pred, linestyle='--',
             label=f'{name} (Test MSE: {test_mse:.2f})')

# 保存对比图
plt.title('Linear Regression Methods Comparison')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(results_dir, 'linear_fits_comparison.png'))
plt.close()
    