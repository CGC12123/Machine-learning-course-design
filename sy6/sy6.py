import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim

from sy6_model import Net

# 获取鸢尾花的数据
iris = datasets.load_iris()
# 切割80%训练和20%的测试数据
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(
    iris.data, iris.target, test_size=0.2)

# 转换为Tensor格式
X_train = torch.Tensor(iris_X_train)
y_train = torch.LongTensor(iris_y_train)
X_test = torch.Tensor(iris_X_test)
y_test = torch.LongTensor(iris_y_test)

# 实例化神经网络模型
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 在测试集上进行预测
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

# 打印输出：实际分类、预测分类、准确率（保留两位有效位）
accuracy = torch.sum(predicted == y_test).item() / len(y_test)
print("实际分类：", y_test)
print("预测分类：", predicted)
print(f"准确率：{accuracy:.2%}")