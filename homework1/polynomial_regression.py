import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from linear_regression import LinearRegression  # 新增导入自定义类

# 在 PolynomialFeatures 导入之后添加数据加载
# 加载训练集和测试集
# 修改数据加载部分（跳过标题行）
train_data = pd.read_csv('e:/TESTpy/PRML/homework1/data/train.txt', sep='\t', skiprows=1, header=None, names=['x_new', 'y_new_complex'])
test_data = pd.read_csv('e:/TESTpy/PRML/homework1/data/test.txt', sep='\t', skiprows=1, header=None, names=['x_new', 'y_new_complex'])

# 保持后续处理代码不变
X_train = train_data['x_new'].values.reshape(-1, 1)
y_train = train_data['y_new_complex'].values
X_test = test_data.iloc[:, 0].values.reshape(-1, 1)
y_test = test_data.iloc[:, 1].values
# === 数据加载结束 ===

# 原有多项式回归分析代码
degrees = range(1, 6)
poly_results = []

for degree in degrees:
    # 多项式特征转换
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 训练模型
    model = LinearRegression(method='ols')
    model.fit(X_train_poly[:,1:], y_train)  # 跳过偏置项
    
    # 预测与评估
    train_pred = model.predict(X_train_poly[:,1:])
    test_pred = model.predict(X_test_poly[:,1:])
    
    poly_results.append({
        'degree': degree,
        'train_mse': mean_squared_error(y_train, train_pred),
        'test_mse': mean_squared_error(y_test, test_pred),
        'model': model
    })

# 结果可视化
plt.figure(figsize=(12, 6))
plt.plot(degrees, [res['train_mse'] for res in poly_results], 'bo-', label='Train MSE')
plt.plot(degrees, [res['test_mse'] for res in poly_results], 'rs--', label='Test MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('MSE')
plt.xticks(degrees)
plt.legend()
plt.title('Polynomial Regression Performance')
plt.savefig('e:/TESTpy/PRML/homework1/results/poly_mse_comparison.png')
plt.close()

# 最佳模型展示
best_model = sorted(poly_results, key=lambda x: x['test_mse'])[0]
print(f"\n最佳多项式模型 (degree={best_model['degree']}):")
print(f"训练MSE: {best_model['train_mse']:.4f}, 测试MSE: {best_model['test_mse']:.4f}")

# 预测结果可视化
plt.figure(figsize=(10, 6))
x_plot = np.linspace(X_test.min(), X_test.max(), 100).reshape(-1,1)
poly = PolynomialFeatures(degree=best_model['degree'])
X_plot_poly = poly.fit_transform(x_plot)

plt.scatter(X_test, y_test, color='gray', label='True')
plt.plot(x_plot, best_model['model'].predict(X_plot_poly[:,1:]), 'r-', lw=2, label='Predicted')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f'Polynomial Regression (Degree {best_model["degree"]})')
plt.savefig('e:/TESTpy/PRML/homework1/results/best_poly_fit.png')
plt.close()