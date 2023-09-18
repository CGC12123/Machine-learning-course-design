from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()

# 得到数据集目标
X = iris.data
y = iris.target

# 将数据集拆分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 实现k-近邻算法，且k=4
knn = KNeighborsClassifier(n_neighbors=4)

# 在训练集上训练k近邻分类器
knn.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = knn.predict(X_test)

# 输出实际分类、预测分类和准确率
for actual, predicted in zip(y_test, y_pred):
    print("Actual:", actual, "Predicted:", predicted)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f" % accuracy)