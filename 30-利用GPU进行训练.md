# 30-利用GPU进行训练

只需要在原始代码上进行小幅度修改，有两张方法

- 方法1：找到网络模型、数据(输入 & 输出、标注)、损失函数，对其调用`.cuda()`
  - 在pycharm项目下新建python脚本train_gpu_1.py

```python
# train.py
import torchvision
from torch.utils.datya import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time


# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 训练集没有cuda方法，只有训练过程中的数据才有cuda方法
# 查看数据集大小
train_data.size = len(train_data)
test_data.size = len(test_data)
print("训练数据集的长度为：{}".format(train_data.size))
# 训练数据集的长度为：50000
# 将光标放在上一行代码的尾部，按下Ctrl+D,实现对上一行代码的复制
print("训练数据集的长度为：{}".format(train_data.size))
# 训练数据集的长度为：10000

# 利用DataLoader来加载数据集
train_dataLoader = DataLoder(train_data, batch_size=64)
test_datalaoder = DataLoder(test_data, batch_size=64)

# 创建网路模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
        	nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
   
tudui = Tudui()
if torch.cuda.is_available():# 判断cuda是否可用(可以实现在cpu和gpu都可以跑)
	tudui = tudui.cuda()# 网络模型转移到cuda上面


# 损失函数
loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
	loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 0.01# 方便修改，也可以写成1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate, )
# 优化器没有cuda方法

# 设置训练网络的一些参数
total_train_step = 0	# 记录训练次数
total_test_step = 0		# 记录测试次数
epoch = 10		# 训练次数

# 添加tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()# 记录时间

for i in range(epoch):
    print("------------------第 {} 轮训练开始-----------------".format(i+1))
    # 训练步骤开始
    tudui.train()
		# 设置模型进入训练模式，只对特定的模型有作用，比如Dropout, BatchNorm
    for data in train_dataLoader:
        imgs, targets = data
        if torch.cuda.is_available():
          img = imgs.cuda()
          target = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        # 不让他每次都打印(逢100才打印)
        if total_traib_step % 100 == 0:
          end_time = time.time()
          print(end_time - start_time)
          print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
          
          writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    # 测试步骤开始
    tudui.eval()
    # 设置模型进入验证模式，只对特定的模型有作用，比如Dropout, BatchNorm
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
      for data in test_dataloader:
        imgs,  targets = data
        if torch.cuda.is_available():
          imgs = imgs.cuda()
          targets = targets.cuda()
        outputs = tudui(imgs)
        loss = loss_fn(outputs)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step +  1
    # 保存每一轮训练的成果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close() 
        
```



- 方法2：找到网络模型、数据(输入 & 输出、标注)、损失函数，对其调用`.to(device)`
- 要先定义设备：`Device = torch.device("cpu")` 或者`Device = torch.device("cuda:0")`(有多个显卡：cuda:x)
  - 在pycharm项目下新建python脚本train_gpu_2.py

```python
# train.py
import torchvision
from torch.utils.datya import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import time

# 定义训练设备
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 准备数据集
train_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=True, transform=torchvision.transforms.ToTensor(), download=True)
test_data = torchvision.datasets.CIFAR10(root="./dataset/CIFAR10", train=False, transform=torchvision.transforms.ToTensor(), download=True)
# 训练集没有cuda方法，只有训练过程中的数据才有cuda方法
# 查看数据集大小
train_data.size = len(train_data)
test_data.size = len(test_data)
print("训练数据集的长度为：{}".format(train_data.size))
# 训练数据集的长度为：50000
# 将光标放在上一行代码的尾部，按下Ctrl+D,实现对上一行代码的复制
print("训练数据集的长度为：{}".format(train_data.size))
# 训练数据集的长度为：10000

# 利用DataLoader来加载数据集
train_dataLoader = DataLoder(train_data, batch_size=64)
test_datalaoder = DataLoder(test_data, batch_size=64)

# 创建网路模型
class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.model = nn.Sequential(
        	nn.Conv2d(3, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, 1, 2),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64*4*4, 64),
            nn.Linear(64, 10),
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
   
tudui = Tudui()
tudui = tudui.to(device)


# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)

# 优化器
learning_rate = 0.01# 方便修改，也可以写成1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate, )
# 优化器没有cuda方法

# 设置训练网络的一些参数
total_train_step = 0	# 记录训练次数
total_test_step = 0		# 记录测试次数
epoch = 10		# 训练次数

# 添加tensorboard
writer = SummaryWriter("./logs_train")
start_time = time.time()# 记录时间

for i in range(epoch):
    print("------------------第 {} 轮训练开始-----------------".format(i+1))
    # 训练步骤开始
    tudui.train()
		# 设置模型进入训练模式，只对特定的模型有作用，比如Dropout, BatchNorm
    for data in train_dataLoader:
        imgs, targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step = total_train_step + 1
        # 不让他每次都打印(逢100才打印)
        if total_traib_step % 100 == 0:
          end_time = time.time()
          print(end_time - start_time)
          print("训练次数：{}，Loss:{}".format(total_train_step, loss.item()))
          
          writer.add_scalar("train_loss", loss.item(), total_train_step)
        
    # 测试步骤开始
    tudui.eval()
    # 设置模型进入验证模式，只对特定的模型有作用，比如Dropout, BatchNorm
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
      for data in test_dataloader:
        imgs,  targets = data
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = tudui(imgs)
        loss = loss_fn(outputs)
        total_test_loss = total_test_loss + loss.item()
        accuracy = (outputs.argmax(1) == targets).sum()
        total_accuracy = total_accuracy + accuracy
    print("整体测试集上的Loss:{}".format(total_test_loss))
    print("整体测试集上的正确率:{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
    total_test_step = total_test_step +  1
    # 保存每一轮训练的成果
    torch.save(tudui, "tudui_{}.pth".format(i))
    print("模型已保存")

writer.close() 
        
```
