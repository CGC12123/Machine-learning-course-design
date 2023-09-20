import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

from sy6_model import Net

# 获取鸢尾花数据
iris = datasets.load_iris()
# 切割80%训练和20%的测试数据
X_train, X_test, y_train, y_test = train_test_split(iris.data,
                                                    iris.target,
                                                    test_size=0.2,
                                                    random_state=42)

# 转换为Tensor格式
X_train = torch.Tensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.Tensor(X_test)
y_test = torch.LongTensor(y_test)

# 实例化神经网络模型
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss() # 交叉熵
optimizer = optim.SGD(model.parameters(), lr=0.01,
                      weight_decay=0.001)  # 添加L2正则化项

# 定义存储训练过程中的损失和准确率的列表
train_loss_history = []
train_acc_history = []

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    optimizer.zero_grad() # 优化器 optimizer 的梯度缓存清零
    loss.backward()
    optimizer.step()

    # 在训练过程中计算准确率
    _, predicted = torch.max(outputs.data, 1) # 找寻最大概率标签
    accuracy = torch.sum(predicted == y_train).item() / len(y_train)

    # 记录训练过程中的损失和准确率
    train_loss_history.append(loss.item())
    train_acc_history.append(accuracy)

    if (epoch + 1) % 10 == 0:
        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}"
        )

# 在测试集上进行预测
with torch.no_grad():
    model.eval() # 切换为评估模式
    outputs = model(X_test)
    _, predicted = torch.max(outputs.data, 1)

# 打印输出：实际分类、预测分类、准确率（保留两位有效位）
accuracy = torch.sum(predicted == y_test).item() / len(y_test)
print("实际分类：", y_test)
print("预测分类：", predicted)
print(f"准确率：{accuracy:.2%}")

# 绘制训练过程中的损失和准确率曲线
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), train_loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), train_acc_history)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()