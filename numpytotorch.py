'''
numpy提供一个n维数组对象及相关操作函数
numpy是一个用于科学计算的通用框架
可以通过使用numpy操作手动实现网络的前后向传播
'''
import numpy as np
import math

x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)

a = np.random.rand()
b = np.random.rand()
c = np.random.rand()
d = np.random.rand()

lr = 1e-6

for epch in range(50000):
    y_hat = a + b*x + c*x ** 2 + d * x ** 3

    loss = np.square(y_hat - y).sum()
    if epch % 100 == 0:
        print(epch, loss)

    d_y_hat = 2.0 * (y_hat - y)
    d_a = d_y_hat.sum()
    d_b = (d_y_hat * x).sum()
    d_c = (d_y_hat * x ** 2).sum()
    d_d = (d_y_hat * x ** 3).sum()

    a = a - lr * d_a
    b = b - lr * d_b
    c = c - lr * d_c
    d = d - lr * d_d

print(f"Result: y={a} + {b} + {c} x^2 + {d} x^3")



'''
PyTorch
PyTorch tensor即张量 是一个n维数组
PyTorch 提供了许多对这些张量进行操作的函数
张量可以跟踪计算图和梯度 也可以作为科学计算的通用工具
PyTorch可以利用GPU来加速其数值计算
'''
import torch
import math


dtype = torch.float
device = torch.device("cpu")
'''
若在GPU上
使用
device = torch.device("cuda:0")
'''

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)

a = torch.randn((), device = device, dtype = dtype, requires_grad = True)
b = torch.randn((), device = device, dtype = dtype, requires_grad = True)
c = torch.randn((), device = device, dtype = dtype, requires_grad = True)
d = torch.randn((), device = device, dtype = dtype, requires_grad = True)

lr = 1e-6
for epch in range(5000):
    y_hat = a + b * x + c * x ** 2 + d * x ** 3
    loss = (y_hat - y).pow(2).sum()
    if epch % 100 == 0:
        print(epch, loss.item())
    
loss.backward()

with torch.no_grad():
    a = a - lr * a.grad
    b = b - lr * b.grad
    c = c - lr * c.grad
    d = d - lr * d.grad

    a.grad = None
    b.grad = None
    c.grad = None
    d.grad = None

print(f"Result: y = {a.item()} + {b.item()} x + {c.item()} x^2 + {d.item()} x^3")




'''
定义autograd函数
'''
import torch
import math

class LegendrePolynomial3(torch.autograd.Function):

# '''
# 静态方法谁都可以调用
# 不需要实例化
# '''

    @staticmethod
    def forward(qwy, input):
        qwy.save_for_backward(input)
        return 0.5 * (5 * input ** 3 - 3 * input)

    @staticmethod
    def backward(qwy, grad_output):
        input, = qwy.saved_tensors
        return grad_output * 1.5 * (5 * input ** 2 - 1)

dtype = torch.float
device = torch.device("cpu")
#device = torch.device("cuda:0")

x = torch.linspace(-math.pi, math.pi, 2000, device = device, dtype = dtype)
y = torch.sin(x)

a = torch.full((), 0.0, device = device, dtype = dtype, requires_grad = True)
b = torch.full((), -1.0, device = device, dtype = dtype, requires_grad = True)
c = torch.full((), 0.0, device = device, dtype = dtype, requires_grad = True)
d = torch.full((), 0.3, device = device, dtype = dtype, requires_grad = True)

lr = 5e-6
for epch in range(5000):
    P3 = LegendrePolynomial3.apply
    y_hat = a + b * P3(c + d * x)
    loss = (y_hat - y).pow(2).sum()
    
    if epch % 100 == 0:
        print(epch, loss.item())

    loss.backward()

    with torch.no_grad():
        a -= lr * a.grad
        b -= lr * b.grad
        c -= lr * c.grad
        d -= lr * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None

print(f"Result: y = {a.item()}, {b.item()} * P3({c.item()} + {d.item()} x)")



'''
nn模块
'''
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0,1)
)

loss_fn = torch.nn.MSELoss(reduction = 'sum')

lr = 1e-6
for epch in range(2000):
    y_hat = model(xx)
    loss = loss_fn(y_hat, y)
    if epch % 100 == 0:
        print(epch, loss.item())
    
    model.zero_grad()
    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad

linear_layer = model[0]
print(f'Result: y = {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3')





'''
PyTorch优化
'''
import torch
import math

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

p = torch.tensor([1, 2, 3])
xx = x.unsqueeze(-1).pow(p)

model = torch.nn.Sequential(
    torch.nn.Linear(3, 1),
    torch.nn.Flatten(0, 1)
)

loss_fn = torch.nn.MSELoss(reduction='sum')

lr = 1e-3
optimizer = torch.optim.RMSprop(model.parameters(), lr = lr)

for epch in range(5000):
    y_hat = model(xx)

    loss = loss_fn(y_hat, y)
    if epch % 100 == 0:
        print(epch, loss.item())

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()

    linear_layer = model[0]

    print(f"Result: y =  {linear_layer.bias.item()} + {linear_layer.weight[:, 0].item()} x + {linear_layer.weight[:, 1].item()} x^2 + {linear_layer.weight[:, 2].item()} x^3")



'''
自定义神经网络    
'''
import math
from pickletools import optimize
from turtle import forward
import torch

class Polynomial3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(()))
        self.b = torch.nn.Parameter(torch.randn(()))
        self.c = torch.nn.Parameter(torch.randn(()))
        self.d = torch.nn.Parameter(torch.randn(()))

    def forward(self, x):
        return self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3

    def string(self):
        return f"y = {self.a.item()} + {self.b.item()} x + {self.c.item()} x^2 + {self.d.item()} x^3"

x = torch.linspace(-math.pi, math.pi, 2000)
y = torch.sin(x)

model = Polynomial3()

criterion = torch.nn.MSELoss(reduction = 'sum')
optimizer = torch.optim.SGD(model.parameters(), lr = 1e-6)

for epch in range(2000):
    
    y_hat = model(x)

    loss = criterion(y_hat, y)
    if epch % 100 == 0:
        print(epch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Result: {model.string()}")