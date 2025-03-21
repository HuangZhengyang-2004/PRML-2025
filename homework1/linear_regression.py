import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, method='ols', max_iter=1000, alpha=0.01, tol=1e-6):
        self.method = method
        self.max_iter = max_iter
        self.alpha = alpha  # 学习率
        self.tol = tol      # 收敛阈值
        self.weights = None

    def fit(self, X, y):
        X = np.c_[np.ones(len(X)), X]  # 添加偏置项
        if self.method == 'ols':
            # 最小二乘法闭式解
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.method == 'gd':
            # 梯度下降实现
            self.weights = np.zeros(X.shape[1])
            for _ in range(self.max_iter):
                grad = X.T @ (X @ self.weights - y) / len(X)
                self.weights -= self.alpha * grad
                if np.linalg.norm(grad) < self.tol:
                    break
        elif self.method == 'newton':
            # 牛顿法实现
            self.weights = np.zeros(X.shape[1])
            for _ in range(self.max_iter):
                error = X @ self.weights - y
                grad = X.T @ error / len(X)
                hessian = X.T @ X / len(X)
                delta = np.linalg.inv(hessian) @ grad
                self.weights -= delta
                if np.linalg.norm(delta) < self.tol:
                    break
        return self

    def predict(self, X):
        X = np.c_[np.ones(len(X)), X]
        return X @ self.weights

# 数据加载
train_data = pd.read_csv('e:/TESTpy/PRML/homework1/data/train.txt', sep='\t')
test_data = pd.read_csv('e:/TESTpy/PRML/homework1/data/test.txt', sep='\t')

X_train, y_train = train_data['x_new'].values.reshape(-1,1), train_data['y_new_complex']
X_test, y_test = test_data['x_new'].values.reshape(-1,1), test_data['y_new_complex']

# 模型训练与评估
methods = ['ols', 'gd', 'newton']
results = {}

for method in methods:
    model = LinearRegression(method=method)
    model.fit(X_train, y_train)
    
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    results[method] = {
        'train_mse': mean_squared_error(y_train, train_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'coef': model.weights
    }

# 结果输出
print("线性模型结果对比:")
for method, res in results.items():
    print(f"\n{method.upper()}方法:")
    print(f"训练MSE: {res['train_mse']:.4f}, 测试MSE: {res['test_mse']:.4f}")
    print(f"模型参数: w0={res['coef'][0]:.4f}, w1={res['coef'][1]:.4f}")

# 新增可视化代码
plt.figure(figsize=(12, 6))

# 绘制训练数据散点
plt.scatter(X_train, y_train, color='gray', alpha=0.5, label='Training Data')

# 生成预测用的连续数据
x_plot = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)

# 绘制各方法拟合线
colors = ['blue', 'green', 'red']
for (method, res), color in zip(results.items(), colors):
    y_plot = res['coef'][0] + res['coef'][1] * x_plot
    plt.plot(x_plot, y_plot, color=color, 
             linewidth=2, 
             label=f'{method.upper()} Fit')

plt.xlabel('x_new')
plt.ylabel('y_new_complex')
plt.title('Linear Regression Methods Comparison')
plt.legend()
plt.grid(True)

# 保存图片
plt.savefig('e:/TESTpy/PRML/homework1/results/linear_fit_comparison.png')
plt.close()