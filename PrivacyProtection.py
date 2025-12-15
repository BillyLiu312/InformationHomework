import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 模拟数据
np.random.seed(42)
data_size = 1000
data = {
    'age': np.random.randint(18, 80, data_size),  # 年龄
    'cholesterol': np.random.randint(150, 300, data_size),  # 胆固醇水平
    'blood_pressure': np.random.randint(100, 180, data_size),  # 血压
    'has_disease': np.random.randint(0, 2, data_size)  # 是否患病（0: 否，1: 是）
}

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 数据预处理：处理缺失值和标准化
X = df.drop(columns=['has_disease'])
y = df['has_disease']

# 假设我们有一些缺失值，使用均值填充
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

print(X_train.shape, X_test.shape)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 创建并训练支持向量机模型
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 性能评估
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 添加噪声来模拟差分隐私
def apply_differential_privacy(data, epsilon=1.0):
    noise = np.random.laplace(loc=0.0, scale=1/epsilon, size=data.shape)
    return data + noise

# 在训练数据上应用差分隐私
X_train_privacy = apply_differential_privacy(X_train)
X_test_privacy = apply_differential_privacy(X_test)

# 重新训练模型
model_privacy = SVC(kernel='linear', random_state=42)
model_privacy.fit(X_train_privacy, y_train)

# 预测
y_pred_privacy = model_privacy.predict(X_test_privacy)

# 性能评估
accuracy_privacy = accuracy_score(y_test, y_pred_privacy)
recall_privacy = recall_score(y_test, y_pred_privacy)
f1_privacy = f1_score(y_test, y_pred_privacy)

print(f"Privacy-preserved Accuracy: {accuracy_privacy:.4f}")
print(f"Privacy-preserved Recall: {recall_privacy:.4f}")
print(f"Privacy-preserved F1 Score: {f1_privacy:.4f}")

# 模拟联邦学习中的多个医院
from sklearn.model_selection import train_test_split

# 医院A数据（从原始训练集划分）
X_train_A, _, y_train_A, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=42)
# 医院B数据
X_train_B, _, y_train_B, _ = train_test_split(X_train, y_train, test_size=0.5, random_state=43)

# 医院A模型
model_A = SVC(kernel='linear', random_state=42)
model_A.fit(X_train_A, y_train_A)

# 医院B模型
model_B = SVC(kernel='linear', random_state=43)
model_B.fit(X_train_B, y_train_B)

# 聚合权重（简单平均）
aggregated_coef = (model_A.coef_ + model_B.coef_) / 2  # shape: (1, n_features)
aggregated_intercept = (model_A.intercept_ + model_B.intercept_) / 2  # shape: (1,)

# 手动预测
decision_values = (X_test @ aggregated_coef.T).flatten() + aggregated_intercept
y_pred_aggregated = (decision_values > 0).astype(int)

# 性能评估
accuracy_aggregated = accuracy_score(y_test, y_pred_aggregated)
recall_aggregated = recall_score(y_test, y_pred_aggregated)
f1_aggregated = f1_score(y_test, y_pred_aggregated)

print(f"Federated Learning Accuracy: {accuracy_aggregated:.4f}")
print(f"Federated Learning Recall: {recall_aggregated:.4f}")
print(f"Federated Learning F1 Score: {f1_aggregated:.4f}")
