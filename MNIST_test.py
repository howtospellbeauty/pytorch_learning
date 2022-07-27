import torch as t
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm

#data
transform = transforms.Compose([transforms.ToTensor(), 
transforms.Normalize((0.5,), (0.5,)), ])

mnist_trainset = datasets.MNIST(root='./data', train=True, 
download=True, transform=transform)
train_loader = t.utils.data.DataLoader(mnist_trainset, batch_size=1000, 
shuffle=True)

mnist_testset = datasets.MNIST(root='./data', train=False, 
download=True, transform=transform)
test_loader = t.utils.data.DataLoader(mnist_testset, batch_size=1000, 
shuffle=True)

#model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.Linear1 = nn.Linear(28*28, 100)
        self.Linear2 = nn.Linear(100, 50)
        self.final = nn.Linear(50, 10)
        self.relu = nn.ReLU()

    #convert + flatten : 转化+扁平化
    def forward(self, img):
        x = img.view(-1, 28*28)
        x = self.relu(self.Linear1(x))
        x = self.relu(self.Linear2(x))
        x = self.final(x)
        return x
net = Net()

#Loss Function
cross_el = nn.CrossEntropyLoss()
optimizer = t.optim.Adam(net.parameters(), lr=0.001)
epoch = 2

for epoch in range(epoch):
    net.train()

    for data in tqdm(train_loader):
        x, y = data
        optimizer.zero_grad()
        output = net(x)
        loss = cross_el(output, y)
        loss.backward()
        optimizer.step()

#Evaluating our dataset
correct = 0
total = 0
with t.no_grad():
    for data in test_loader:
        x, y = data
        output = net(x.view(-1, 784))
        for idx, i in enumerate(output):
            if t.argmax(i) == y[idx]:
                correct += 1
            total += 1
    print(f'accuracy:{round(correct/total, 3)}')

#visualization
for i in range(5):
    plt.imshow(x[i].view(28,28))
    plt.title(t.argmax(net(x[i].view(-1, 784))))
    plt.show()
    print(t.argmax(net(x[3].view(-1, 784))))