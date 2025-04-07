import os
import numpy as np
import matplotlib.pyplot as plt

# 定义结果保存目录
results_dir = r'E:\TESTpy\PRML\homework1\results'
os.makedirs(results_dir, exist_ok=True)

# 数据加载
try:
    train_data = np.loadtxt(os.path.join(results_dir, '../data/train.txt'), skiprows=1, delimiter='\t')
    test_data = np.loadtxt(os.path.join(results_dir, '../data/test.txt'), skiprows=1, delimiter='\t')
    X_train, y_train = train_data[:, 0], train_data[:, 1]
    X_test, y_test = test_data[:, 0], test_data[:, 1]
except FileNotFoundError:
    print("数据文件未找到，请检查文件路径。")
    exit(1)

# KNN高斯核回归

def knn_regression_gaussian(x_query, X_train, y_train, k=5, sigma=1.0):
    distances = np.linalg.norm(X_train - x_query, axis=1)
    idx = np.argsort(distances)[:k]
    k_distances = distances[idx]
    k_targets = y_train[idx]

    if sigma == 0:
        return k_targets[np.argmin(k_distances)]
    else:
        weights = np.exp(-(k_distances ** 2) / (2 * sigma ** 2))
        return np.sum(weights * k_targets) / np.sum(weights) if np.sum(weights) != 0 else np.mean(k_targets)

# 网格搜索函数
def grid_search_knn(X_train, y_train, X_test, y_test, k_values, sigma_values):
    best_mse = float('inf')
    best_params = (0, 0)
    optimization_history = []
    mse_grid = np.zeros((len(sigma_values), len(k_values)))

    for k_idx, k in enumerate(k_values):
        for sigma_idx, sigma in enumerate(sigma_values):
            y_pred = np.array([
                knn_regression_gaussian(x, X_train.reshape(-1, 1), y_train, k, sigma)
                for x in X_test
            ])
            current_mse = np.mean((y_pred - y_test) ** 2)
            optimization_history.append(current_mse)
            mse_grid[sigma_idx, k_idx] = current_mse
            if current_mse < best_mse:
                best_mse = current_mse
                best_params = (k, sigma)

    return best_params, mse_grid, optimization_history

# 参数设置
k_values = np.arange(1, 16, 2)
sigma_values = np.linspace(0.1, 2, 20)

# 网格搜索
best_params, mse_grid, optimization_history = grid_search_knn(
    X_train, y_train, X_test, y_test, k_values, sigma_values
)

# 保存MSE网格数据
np.save(os.path.join(results_dir, 'knn_mse_grid.npy'), mse_grid)

# 可视化：参数搜索热力图与收敛曲线
plt.figure(figsize=(14, 6))

plt.subplot(121)
k_grid, sigma_grid = np.meshgrid(k_values, sigma_values)
plt.contourf(k_grid, sigma_grid, mse_grid, cmap='viridis', levels=20)
plt.plot(best_params[0], best_params[1], 'rx', markersize=10, label='Best')
plt.xlabel('k Values')
plt.ylabel('Sigma Values')
plt.colorbar(label='MSE')
plt.title('Grid Search for KNN Regression')
plt.legend()

plt.subplot(122)
plt.plot(optimization_history, 'b-')
plt.xlabel('Parameter Combination Index')
plt.ylabel('MSE')
plt.title('Grid Search Convergence')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'knn_search.png'))
plt.show()

# 可视化：拟合曲线图
x_fit = np.linspace(min(X_test), max(X_test), 300)
y_fit = np.array([
    knn_regression_gaussian(x, X_train.reshape(-1, 1), y_train, best_params[0], best_params[1])
    for x in x_fit
])

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='gray', alpha=0.4, label='Training Data')
plt.scatter(X_test, y_test, color='blue', s=10, label='Test Data')
plt.plot(x_fit, y_fit, color='red', label=f'KNN Fit (k={best_params[0]}, σ={best_params[1]:.2f})')
plt.xlabel('x')
plt.ylabel('y')
plt.title('KNN Regression Fit with Gaussian Kernel')
plt.legend()
plt.savefig(os.path.join(results_dir, 'knn_fit_curve.png'))
plt.show()

print(f"最佳参数组合: k={best_params[0]}, σ={best_params[1]:.2f}")
