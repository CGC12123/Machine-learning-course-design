import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

plt.rcParams['font.sans-serif'] = ['SimHei'] # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 创建PCA对象，将数据降维为2维
pca = PCA(n_components=2)
X_pca_sklearn = pca.fit_transform(X)

fig, axs = plt.subplots(1, 2, figsize=(10, 5)) # 创建子图

# 在第一个子图中绘制样本点的分布（使用sklearn的PCA）
scatter1 = axs[0].scatter(X_pca_sklearn[:, 0], X_pca_sklearn[:, 1], c=y, cmap='viridis')
axs[0].set_xlabel('主成分1')
axs[0].set_ylabel('主成分2')
axs[0].set_title('鸢尾花数据集 - 主成分分析降维 (sklearn)')
colorbar1 = plt.colorbar(scatter1, ax=axs[0], ticks=range(3))
colorbar1.set_label('类别')

#################### 以下为使用numpy ###########################

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算标准化后数据的协方差矩阵
cov_matrix = np.cov(X_scaled.T)

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 选择前两个特征向量对应的特征值最大的成分作为主成分
top_two_idx = np.argsort(eigenvalues)[::-1][:2]
top_two_eigenvectors = eigenvectors[:, top_two_idx]

# 将数据投影到选定的特征向量上
X_pca_numpy = np.dot(X_scaled, top_two_eigenvectors)

# 在第二个子图中绘制样本点的分布（使用NumPy的PCA）
scatter2 = axs[1].scatter(X_pca_numpy[:, 0], X_pca_numpy[:, 1], c=y, cmap='viridis')
axs[1].set_xlabel('主成分1')
axs[1].set_ylabel('主成分2')
axs[1].set_title('鸢尾花数据集 - 主成分分析降维 (numpy)')
colorbar2 = plt.colorbar(scatter2, ax=axs[1], ticks=range(3))
colorbar2.set_label('类别')

plt.tight_layout()
plt.show()