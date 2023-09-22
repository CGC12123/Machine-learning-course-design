import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 检查GPU是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# 设置随机种子
torch.manual_seed(42)

# 下载MNIST数据集
train_dataset = MNIST(root='./data', train=True, download=False, transform=ToTensor())
test_dataset = MNIST(root='./data', train=False, download=False, transform=ToTensor())

# 构建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# 定义卷积神经网络模型
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

# 创建模型实例并将其移动到GPU上
model = CNN().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    for batch_data, batch_labels in train_loader:
        # 将数据移动到GPU上
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # 计算准确率
        _, predicted = torch.max(outputs, 1)
        batch_accuracy = accuracy_score(batch_labels.cpu(), predicted.cpu())
        epoch_accuracy += batch_accuracy

    epoch_loss /= len(train_loader)
    epoch_accuracy /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}")

    # 测试阶段
    model.eval()
    test_accuracy = 0.0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            _, predicted = torch.max(outputs, 1)
            batch_accuracy = accuracy_score(batch_labels.cpu(), predicted.cpu())
            test_accuracy += batch_accuracy

    test_accuracy /= len(test_loader)
    test_accuracies.append(test_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Test Accuracy: {test_accuracy:.4f}")

# 绘制结果曲线
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Value')
plt.title('Training and Testing Results')
plt.legend()
plt.show()