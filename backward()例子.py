'''
import torch
x=torch.arange(start=1,end=5).float()
x.requires_grad_(True)
print(x)
y=x.sum()/5
print(y)
y.backward()
print(x.grad)
x.grad.zero_()
z=torch.dot(x,x.t())
print(z)
z.backward()
print(x.grad)
'''
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# 1. 创建数据
x = torch.tensor([[1.0], [2.0], [3.0], [4.0]], requires_grad=True)
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])

# 2. 定義簡單模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(1, 2)
        self.layer2 = nn.Linear(2, 1)

    def forward(self, x):
        self.hidden_input = x  # 存储第一层的输入
        x = torch.sigmoid(self.layer1(x))
        self.hidden_output=x

        x = self.layer2(x)

        return x

model = SimpleModel()

# 3. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 4. 前向传播
outputs = model(x)
loss = criterion(outputs, y)

# 5. 计算损失
print(f'Loss: {loss.item()}')

# 6. 反向传播
loss.backward()

# 7. 验证梯度
# 打印各層的梯度
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name} grad: {param.grad}')
#8 輸出output
print("output",outputs)
print("model.hidden_output",model.hidden_output)
#手刻梯度

d_loss_d_output = 2 * (outputs - y) / x.size(0)  # 计算损失对输出的梯度
d_output_d_z2 = torch.ones_like(outputs)  # 因为最后一层没有激活函数
d_loss_d_z2 = d_loss_d_output * d_output_d_z2  # 计算损失对 z2 的梯度

'''
print("model.hidden_output[:,0]",model.hidden_output[:,0])
print("model.hidden_output[:,0].unsqueeze(1)",model.hidden_output[:,0].unsqueeze(1))
cw=2 * (outputs - y)*model.hidden_output[:,0].unsqueeze(1)#有unsqueeze可以確保形狀是(4,1)不是原來的(4,)不會因廣播而有差錯
print("cw",torch.mean(cw))
'''
clayer2_weight=torch.matmul(d_loss_d_z2.T,model.hidden_output)
#print("outputs*(1-outputs)*model.hidden_output",outputs*(1-outputs)*(model.hidden_output[:,0]))
#print("outputs*(1-outputs)",outputs*(1-outputs))
print("clayer2_weight",clayer2_weight)

d_loss_d_bias=torch.ones_like(outputs)
clayer2_bias=torch.matmul(d_loss_d_z2.T,d_loss_d_bias)
# 验证与自动计算梯度的差异
print("Difference in layer2.weight.grad:", torch.abs(clayer2_weight - model.layer2.weight.grad).sum().item())
print("Difference in layer2.bias.grad:", torch.abs(clayer2_bias - model.layer2.bias.grad).sum().item())

# 计算隐藏层输出对隐藏层输入的梯度
d_z2_d_a1 = model.layer2.weight.T  # shape: (2, 1)
d_loss_d_a1 = d_loss_d_z2.mm(d_z2_d_a1.T)
print("d_loss_d_z2",d_loss_d_z2)
print("d_loss_d_a1",d_loss_d_a1)
# 计算隐藏层输入对输入层输出的梯度
d_a1_d_z1 = model.hidden_output * (1 - model.hidden_output)

# 计算损失对隐藏层输入的梯度
d_loss_d_z1 = d_loss_d_a1 * d_a1_d_z1

# 计算损失对 layer1 权重的梯度
d_z1_d_w1 = model.hidden_input  # 形状为 ( batch_size,1)
clayer1_weight = d_loss_d_z1.T.mm(d_z1_d_w1) 

print("clayer1_weight",clayer1_weight)
print("Difference in layer1.weight.grad:", torch.abs(clayer1_weight - model.layer1.weight.grad).sum().item())

d_z1_d_bias=torch.ones_like(model.hidden_input)
clayer1_bias = torch.sum(d_loss_d_z1, dim=0, keepdim=True)

#clayer1_bias = d_loss_d_z1.T.mm(d_z1_d_bias) 

print("clayer1_bias",clayer1_bias)
print("Difference in layer1.bias.grad:", torch.abs(clayer1_bias - model.layer1.bias.grad).sum().item())

'''
print("\n第一层的权重和偏置:")
print("权重: ", model.layer1.weight.data)
print("偏置: ", model.layer1.bias.data)

print("\n第二层的权重和偏置:")
print("权重: ", model.layer2.weight.data)
print("偏置: ", model.layer2.bias.data)
'''