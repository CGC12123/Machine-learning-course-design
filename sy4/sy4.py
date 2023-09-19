import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.impute import SimpleImputer

# 加载鸢尾花数据集
iris = datasets.load_iris()
iris_X = iris.data

# 处理缺失值
imputer = SimpleImputer(strategy='mean')
iris_X = imputer.fit_transform(iris_X)

# 实现C均值算法
def c_means(X, k, m, max_iterations=100):
    cmeans = KMeans(n_clusters=k, random_state=0)
    cmeans.n_init = 1  # 禁用随机初始化，使用自定义初始化
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]
    for _ in range(max_iterations):
        distances = pairwise_distances_argmin_min(X, centroids, metric='euclidean')
        labels, _ = distances
        for j in range(k):
            centroids[j] = np.power(np.sum(np.power(distances[1][j] / distances[1], 2 / (m - 1)) * X.T, axis=1) / np.sum(np.power(distances[1][j] / distances[1], 2 / (m - 1))), 1 / m)
    return labels, centroids

# 设置聚类簇的数量（鸢尾花数据集为k=3）
k = 3

# 设置模糊性参数（m > 1）
m = 2

# 应用自定义C均值算法
custom_labels, custom_centroids = c_means(iris_X, k, m)

# 应用scikit-learn的C均值算法
sklearn_cmeans = KMeans(n_clusters=k, random_state=0)
sklearn_labels = sklearn_cmeans.fit_predict(iris_X)
sklearn_centroids = sklearn_cmeans.cluster_centers_

# 绘制结果
plt.figure(figsize=(12, 6))

# 自定义C均值算法
plt.subplot(1, 2, 1)
plt.scatter(iris_X[:, 0], iris_X[:, 1], c=custom_labels)
plt.scatter(custom_centroids[:, 0], custom_centroids[:, 1], marker='X', c='red', s=150)
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title('自定义C均值算法')

# scikit-learn的C均值算法
plt.subplot(1, 2, 2)
plt.scatter(iris_X[:, 0], iris_X[:, 1], c=sklearn_labels)
plt.scatter(sklearn_centroids[:, 0], sklearn_centroids[:, 1], marker='X', c='red', s=150)
plt.xlabel('花萼长度')
plt.ylabel('花萼宽度')
plt.title('scikit-learn的C均值算法')

plt.tight_layout()
plt.show()