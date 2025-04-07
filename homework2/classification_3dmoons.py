import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import os

# 创建结果保存目录
results_dir = "E:/TESTpy/PRML/homework2/results"
os.makedirs(results_dir, exist_ok=True)

# 生成3D make_moons数据
def make_moons_3d(n_samples=500, noise=0.1):
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)
    X = np.vstack([np.column_stack([x, y, z]), np.column_stack([-x, y - 1, -z])])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y

# 生成数据
X_train, y_train = make_moons_3d(n_samples=1000, noise=0.2)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.2)

# 标准化（SVM对尺度敏感）
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 定义分类器
classifiers = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "AdaBoost (DT)": AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=3),
        n_estimators=50,
        random_state=42
    ),
    "SVM (Linear)": SVC(kernel='linear', C=1.0),
    "SVM (RBF)": SVC(kernel='rbf', gamma='scale', C=1.0),
    "SVM (Poly)": SVC(kernel='poly', degree=3, C=1.0)
}

# 训练与评估
results = []
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append((name, acc, f1))
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

# 保存详细结果到 TXT 文件
txt_result_path = os.path.join(results_dir, "detailed_results.txt")
with open(txt_result_path, "w") as f:
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        f.write(f"=== {name} ===\n")
        f.write(f"Accuracy: {acc:.4f}, F1-Score: {f1:.4f}\n")
        f.write("Classification Report:\n")
        f.write(report + "\n")
        f.write("-" * 50 + "\n")

# 打印摘要表格并保存为 CSV
summary_df = pd.DataFrame(results, columns=["Model", "Accuracy", "F1-Score"])
print("\nClassification Performance Summary:")
print(summary_df)
summary_df.to_csv(os.path.join(results_dir, "results_3d_classification.csv"), index=False)

# 动态选择最好和最差模型进行可视化
results.sort(key=lambda x: x[1], reverse=True)
best_model_name = results[0][0]
worst_model_name = results[-1][0]

best_model = classifiers[best_model_name]
y_pred_best = best_model.predict(X_test)

worst_model = classifiers[worst_model_name]
y_pred_worst = worst_model.predict(X_test)

# 可视化
fig = plt.figure(figsize=(18, 6))

ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='viridis')
ax1.set_title('True Labels (Test Set)')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_best, cmap='viridis')
ax2.set_title(f'{best_model_name} Predictions\nAccuracy: {accuracy_score(y_test, y_pred_best):.4f}')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_pred_worst, cmap='viridis')
ax3.set_title(f'{worst_model_name} Predictions\nAccuracy: {accuracy_score(y_test, y_pred_worst):.4f}')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

plt.tight_layout()
plt.savefig(os.path.join(results_dir, "3d_model_comparison.png"))
plt.show()