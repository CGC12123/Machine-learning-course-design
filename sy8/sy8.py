import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 获取手写数字集
digits = load_digits()
# 切割80%训练和20%的测试数据
digits_X_train, digits_X_test, digits_Y_train, digits_Y_test = train_test_split(digits.data, digits.target, test_size=0.2)

# 将数据转换为PyTorch的张量
train_data = torch.from_numpy(digits_X_train).float()
train_labels = torch.from_numpy(digits_Y_train).long()
test_data = torch.from_numpy(digits_X_test).float()
test_labels = torch.from_numpy(digits_Y_test).long()

# 构建数据集和数据加载器
train_dataset = TensorDataset(train_data, train_labels)
test_dataset = TensorDataset(test_data, test_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(0.5)  # 添加Dropout层
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)  # 应用Dropout层
        x = self.fc2(x)
        return x

# 创建模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 1000
losses = []
accuracies = []
for epoch in range(num_epochs):
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    # 每100个epoch输出损失和准确率
    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for batch_data, batch_labels in test_loader:
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

            accuracy = correct / total
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        losses.append(loss.item())
        accuracies.append(accuracy)

# 绘制损失和准确率的图像
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy')

plt.tight_layout()
plt.show()