import torch
import torch.nn as nn
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt
import math

# 定义LeNet-5模型
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.relu1 = nn.ReLU()
        self.avgpool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.relu2 = nn.ReLU()
        self.avgpool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 4 * 4, 120)  # 修改这里的输入维度为 16 * 4 * 4
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avgpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.avgpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x

# 加载LeNet-5模型
model = LeNet5()
model.load_state_dict(torch.load('model/best.pth'))
model.eval()

# 定义图像预处理
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载测试集
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# 遍历测试集进行预测和比较，选择分类准确的五张图像
correctly_classified_samples = []
with torch.no_grad():
    for images, labels in test_loader:
        output = model(images)
        probabilities = torch.softmax(output, dim=1)
        predicted_labels = torch.argmax(probabilities, dim=1)
        if predicted_labels.item() == labels.item():
            correctly_classified_samples.append((images.squeeze(0), labels.item(), predicted_labels.item()))
            if len(correctly_classified_samples) == 5:  # 选择五张图像后停止遍历
                break

# 显示五张分类准确的图像并标出分类信息
num_cols = 5  # 每行显示的图像数量
num_rows = math.ceil(len(correctly_classified_samples) / num_cols)

fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 10))

for i, (image, true_label, predicted_label) in enumerate(correctly_classified_samples):
    if num_rows == 1:
        ax = axes[i % num_cols]
    else:
        ax = axes[i // num_cols, i % num_cols]

    image = transforms.ToPILImage()(image)
    ax.imshow(image, cmap='gray')
    ax.set_title(f'True: {true_label}, Predicted: {predicted_label}')
    ax.axis('off')

plt.tight_layout()
plt.show()