from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# 获取鸢尾花数据
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 数据归一化
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建MLPClassifier模型
model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 在测试集上进行预测
y_pred = model.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率：{accuracy:.2%}")