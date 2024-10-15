import torch
print(torch.__version__)

# pytorch 版本
print(torch.version.cuda)

print(torch.cuda.is_available())


import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 生成数据集
import torch
import torch.nn as nn
import torch.optim as optim

# 创建简单的非线性神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(1, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 训练数据
def generate_data(n_samples=100):
    x = torch.rand(n_samples, 1) * 10  # 随机生成输入 x
    y = torch.sin(x)  # 对应输出 y
    return x, y

# 定义模型、损失函数和优化器
model = SimpleNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
x_train, y_train = generate_data()
x_train_scaled = 10 * x_train  # 创建缩放的输入 10x
y_train_scaled = y_train  # 缩放后输出依然是 y

for epoch in range(10000):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    outputs = model(x_train)
    outputs_scaled = model(x_train_scaled)

    # 损失函数：同时让 f(x) = y 和 f(10x) = y
    loss = criterion(outputs, y_train) + criterion(outputs_scaled, y_train_scaled)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch+1}/1000], Loss: {loss.item():.4f}')

# 测试和可视化结果

model.eval()
with torch.no_grad():
    y_pred = model(x_train)
    y_pred_scaled = model(x_train_scaled)
    y_pred_scaled_100 =  model(10 * x_train_scaled)

# 绘制结果
plt.scatter(x_train.numpy(), y_train.numpy(), label='True y = x^2', color='blue')
plt.scatter(x_train.numpy(), y_pred.numpy(), label='Predicted', color='red')
plt.scatter(x_train_scaled.numpy(), y_train_scaled.numpy(), label='True y = x^2', color='yellow')
plt.scatter(x_train_scaled.numpy()*10, y_pred_scaled_100.numpy(), label='Predicted', color='green')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Fitting y = x^2 using a Neural Network')
plt.show()