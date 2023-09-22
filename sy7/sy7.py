import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.preprocessing import StandardScaler

# 获取乳腺癌数据集
cancers = load_breast_cancer()

# 切割80%训练和20%的测试数据
cancers_X_train, cancers_X_test, cancers_Y_train, cancers_Y_test = train_test_split(cancers.data, cancers.target, test_size=0.2)

# 特征缩放
scaler = StandardScaler()
cancers_X_train = scaler.fit_transform(cancers_X_train)
cancers_X_test = scaler.transform(cancers_X_test)

# 转换数据为PyTorch张量
X_train = torch.from_numpy(cancers_X_train).float()
Y_train = torch.from_numpy(cancers_Y_train).long()
X_test = torch.from_numpy(cancers_X_test).float()
Y_test = torch.from_numpy(cancers_Y_test).long()

# 神经网络模型
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 设置超参数
input_size = cancers.data.shape[1]
hidden_size = 100
num_classes = 2
learning_rate = 0.001
num_epochs = 100

# 初始化神经网络模型
model = NeuralNetwork(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    # 前向传播和计算损失
    outputs = model(X_train)
    loss = criterion(outputs, Y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 在测试集上进行预测
with torch.no_grad():
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

# 打印输出：实际分类、预测分类、准确率（保留两位有效位）
def print_classification_results(actual, predicted):
    accuracy = accuracy_score(actual, predicted)
    recall = recall_score(actual, predicted)
    
    print("实际分类：", actual)
    print("预测分类：", predicted)
    print("准确率：{:.2%}".format(accuracy))
    print("准确率：{:.2%}".format(recall))

# 转换预测结果为numpy数组
predicted = predicted.numpy()

# 打印分类结果
print_classification_results(cancers_Y_test, predicted)