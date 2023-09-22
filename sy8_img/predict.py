import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

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

# 加载待预测的图像
image = Image.open('img/5.png')
image = transform(image).unsqueeze(0)  # 增加一维表示batch_size

# 进行预测
with torch.no_grad():
    output = model(image)

# 获取预测结果
probabilities = torch.softmax(output, dim=1)
predicted_label = torch.argmax(probabilities, dim=1).item()

print(f"Predicted Label: {predicted_label}")