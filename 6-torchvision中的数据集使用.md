# 6-torchvision中的数据集使用

<img src="https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112212.png" alt="image-20251025180732577" style="zoom:50%;" />

本节课将使用CIFAR数据集，里面会有10类标签，每个类别有6000张图像，训练集图片共5w张，测试集图像为1w张，主要用于物体识别

在pycharm的learn_pytorch项目中新建python文件P10_dataset_transform

![image-20251025180950765](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112271.png)

- touchvision.models里面会提供一些常用的神经网络，有些已经预训练好了，后面神经网络会用到

  ![image-20251025181116928](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112135.png)

- touchvision.transforms上节已经讲过

- 其他的（比如touchvision.io、touchvision.ops）用的不多

在learn_pytorch项目下新建python文档`P10_dataset_trans`

如何使用torchvision提供的标准数据集（点击上上上图中的CIFAR数据集），参数都比较相似

![image-20251025181525231](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112416.png)

- root：数据集的位置
- train：如果为true,则创建数据集为训练集；如果为false,则创建数据集为测试集
- transform：对数据集数据进行变换
- target_transform：对数据集标签进行变换
- download：如果为true，则会自动从网上下载数据集，并放在root路径下，比较方便；如果为false，则不会自动从网上下载数据集

```python
import torchvision

train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, download=True)
# download可以一直设置为True
test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False, download=True)
# 可以将光标在括号里，按下Ctrl+P键，可以看到参数提示
# 会在P10_dataset_transform中出现dataset文件夹
```

运行后可以在下方的Run界面看到数据下载来源，下载到的文件路径和下载进度

![image-20251025182353430](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112223.png)

如果下载比较慢，可以将下载连接（http格式）放到迅雷中进行下载,将下载好压缩包传到dataset的文件夹里（文件名还是dataset），再次运行代码，会直接对下载好的压缩包进行解压

![image-20251025184957514](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112246.png)

> 如果没有给出下载连接，可以按住Ctrl点击代码中的CIFAR10，可以在说明文档中url看到下载连接
>
> ![image-20251025185154195](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112969.png)

下载完成后，可以在dataset中看到下载内容，首先它是先下载tar.gz压缩包，在对其解压为python文件，这个数据集比较小只有100+MB（COCO数据集有30+GiB）

![image-20251025182644207](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112846.png)

```python
# 查看dataset的第一个数据集
print(test_set[0])
# 可以看到数据集的类型
```

![image-20251025182905925](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112915.png)

- 3：某个标签对应的索引值，这边是指cat

![image-20251025183138108](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112999.png)

----

```python
# 打印所有标签
print(test_set.classes)

# 根据print(test_set[0]) 的输出，可以将代码改为
img, target = test_set[0]
# 如果数据集已经下载好，则再次运行python文件则不会进行下载，可以download可以一直设置为True
print(img)
print(target)
print(test_set.classes[target])
# 查看图片（由于图片尺寸为32*32， 图像显示的很小）
img.show()
```

```python
# 将图像转为tensor类型，输入给pytorch
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 设置torchvision.datasets.CIFAR10的transform属性
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transform, download=True)

test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,
transform=dataset_transform, download=True)

print(test_set[0])
```

输出是tensor数据类型

![image-20251025184031112](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112079.png)

```python
# 将其进行tensor显示
from torch.utils.tensorboard import SummaryWriter
import torchvision

dataset_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
# 设置torchvision.datasets.CIFAR10的transform属性
train_set = torchvision.datasets.CIFAR10(root="./dataset",train=True, transform=dataset_transform, download=True)

test_set = torchvision.datasets.CIFAR10(root="./dataset",train=False,
transform=dataset_transform, download=True)

writer = SummaryWriter("p10")
for i in range(10):
    img, target = test_set[i]
    writer.add_image("test_set", img, i)
writer.close()
```

可以看到坐标Project页面的logs文件夹里会生成一个新的log文件，然后在Terminal中输入`tensorboard --logdir="p10"`,点击显示的端口，可以在显示的页面中看到图片

![image-20251025184606777](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112302.png)

6-DataLoader的使用

![image-20251025185522196](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112363.png)

- dataset：只有数据集的位置和数据集当中的索引位置的数，可以看成是一落扑克牌，可以通过某种方法知道第几张扑克牌是什么
- dataloader:是加载器，将数据集加载到神经网络中，相当是手的作用，至于取多少数据，怎么取的过程取决于dataloader的设置

通过说明文档了解dataloader

![image-20251025185914961](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112492.png)

- dataloader是在torch.utils.data.DataLoader包下面
- dataset:数据集的位置以及第X个数据是什么
- 其他参数都有默认值，使用过程只需要设置常见参数的值
  - batch_size:每次从一落牌中抓batch_size张
  - shuffle：每次抓牌前是否需要打乱排序（我们喜欢将其设置为True）
  - num_workers:加载数据集是多进程（加载速度会比较快），还是单进程，一般设置为0（主线程）
    - windows环境下有时会出现问题，windows条件下需要设置为0，不然会报错BrokenPipError
  - drop_last:比如有100张牌，batch_size为false,最后一张牌仍然需要取出

在pycharm的learn_pytorch项目中新建python文件datalaoder

```python
import torchvision
from torch.utils.data import DataLoader

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.ToTensor())

test_loader = DataLoader(dataset=test_Data, batch_size=4, shuffle=True, num_workers=0, drop_last=False)

# 查看test_data中的第一张图片及target
# 可以按住Ctrl点击代码钟大哥CIFAR，查看源码中__getitem__的return img, target
# 返回图片和标签
img, target = test_data[0]
print(img.shape)# 查看图片大小
# torch.Size([3, 32, 32]):3通道，尺寸32 * 32
print(target)
# 数据集对应的标签值索引为3， 对应的是cat
```

![image-20251025221758790](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112946.png)

dataLoader也是讲batch_size采样后的4个image 和对应的target打包后分别存储在一起，用变量imgs和targets来进行存储

```python
for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
```

![image-20251025222321973](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112135.png)

上图的：

- 4表示4张3通道，尺寸32 * 32的图片
- 【2， 3， 6， 8】：数据集对应的标签值索引（sample从不同的标签数据集随机采样）

![image-20251025222536006](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112126.png)

```python
# 展示图片
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_Data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)


writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data", imgs, step)
    step = step + 1

writer.close()
   
```

可以看到坐标Project页面的logs文件夹里会生成一个新的log文件，然后在Terminal中输入`tensorboard --logdir="dataloader"`,点击显示的端口，可以在显示的页面中看到图片

![image-20251025223046431](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112344.png)

此时的step表示第几次从数据集中随机采样数据（64张图片），最后一步是没有64张图片，原因在于drop_last=True

```python
# 展示图片
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False, num_workers=0, drop_last=False)

writer = SummaryWriter("dataloader")
step = 0
for data in test_loader:
    imgs, targets = data
    writer.add_images("test_data_drop_last", imgs, step)
    # 修改DataLoader的参数时，一定要将title进行修改
    step = step + 1


writer.close()
```

最后一次采样时的对比，drop_last=False，他将最后的16张进行舍弃，此时的step只有155，而不是156

![image-20251025223409383](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112618.png)

```python
# for循环将整个数据遍历完，则经历了一次shuffle，如果shuffle=true,那么下次epoch采样前就会将数据进行重新打散
# shuffle=Flase
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=Falase, num_workers=0, drop_last=False)


writer = SummaryWriter("dataloader")
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epochv: {}".format(epoch), imgs, step)
        # 修改DataLoader的参数时，一定要将title进行修改
        step = step + 1


writer.close()
# 每一step 采样的图片完全一样
```

> # 修改DataLoader的参数时，一定要将title进行修改

![image-20251025224049755](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112965.png)

```python
# shuffle=True
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# 准备的测试数据集
test_data = torchvision.datasets.CIFAR10("./dataset", train=False, transform=torchvision.transforms.ToTensor())

test_loader = DataLoader(dataset=test_Data, batch_size=64, shuffle=True, num_workers=0, drop_last=False)

img, target = test_data[0]
print(img.shape)# 查看图片大小
print(target)

writer = SummaryWriter("dataloader")
step = 0
for epoch in range(2):
    for data in test_loader:
        imgs, targets = data
        writer.add_images("Epochx: {}".format(epoch), imgs, step)
        # 修改DataLoader的参数时，一定要将title进行修改
        step = step + 1


writer.close()
# 每一step 采样的图片完全不一样
```

![image-20251025224103551](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062112047.png)