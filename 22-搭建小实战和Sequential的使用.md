# 22-搭建小实战和Sequential的使用

- Sequential：将网络结构放在Sequential()里面，然后model(input),结构里面的顺序是先对input执行Conv2d(1, 20, 5)，再执行ReLU(),接着执行Conv2d(20, 64, 5)， 最后执行ReLU()；

- 好处：代码写起来比较简洁，也易懂

![image-20251028231659440](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104433.png)

CIFAR10 模型

![image-20251029193726492](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104476.png)

- 经过最大池化，移动格数=kernel_size=2,横向&纵向都是减半
- 第一次卷积后通道数变为32，由于卷积加入了padding，32也是调参调出来的
  - 卷积核的维度【5，5，3， 32】采用 32个卷积核，大小为5*5，3是输入图像的通道数，32是输出图像的通道数，每一个卷积核的尺寸为5x5x3（最后的3就是原图的rgb通道数3），每一个卷积核的每一层（5x5）与原图的每一层（32x32）相乘，然后将得到的**三层**结果对应位置叠加（算术求和），就得到这个点对应的卷积结果了。所有的点卷积完成之后则可以得到一张新的feature map
- output设置为10：相当于是对MNIST数据集进行识别

在pycharm项目下新建python脚本nn_Seq,里面会涉及到计算 padding=2， stride=1(stride值过大，padding的会很多，影响卷积效果)

![image-20251029202809730](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104498.png)

> 假设：
>
> - 输入宽度：$W_{in}$
> - 卷积核宽度：$K$
> - 步长：$S$
>
> 那么卷积核每次滑动的范围是：
> $$
> \text{第一个位置：覆盖 [0, K-1]} \\
> \text{第二个位置：覆盖 [S, S + K - 1]} \\
> \text{第三个位置：覆盖 [2S, 2S + K - 1]} \\
> \ldots
> $$
> 卷积核能滑动的最后一个位置是刚好不超过输入的最右端，即：
> $$
> nS + (K - 1) < W_{in}
> $$
> 所以最大整数 $n$ 满足：
> $$
> n = \left\lfloor \frac{W_{in} - (K-1)}{S} \right\rfloor
> $$
> 输出宽度是位置个数 = $n + 1$，于是：
> $$
> \boxed{W_{out} = \left\lfloor \frac{W_{in} - (K-1)}{S} \right\rfloor + 1}
> $$
>
> ------
>
> ## 🧱 三、考虑 Padding（补零）
>
> 如果我们在输入两边各补上 `padding` 个像素，总共多了 `2 × padding` 的宽度。
>
> 于是有效输入宽度变成：
> $$
> W_{in}^{\text{eff}} = W_{in} + 2 \times padding
> $$
> 代入原公式：
> $$
> W_{out} = \left\lfloor \frac{W_{in} + 2 \times padding - (K-1)}{S} \right\rfloor + 1
> $$
>
> ------
>
> ## 🧩 四、考虑 Dilation（膨胀卷积）
>
> 当 dilation > 1 时，卷积核内部元素之间会“插空”，
>  使得卷积核的**有效感受野（覆盖范围）**变大。
>
> 有效卷积核宽度：
> $$
> K_{\text{eff}} = dilation \times (K - 1) + 1
> $$
> 于是：
> $$
> W_{out} = \left\lfloor \frac{W_{in} + 2 \times padding - K_{\text{eff}}}{S} \right\rfloor + 1
> $$
> 展开 $K_{\text{eff}}$：
> $$
> \boxed{
> W_{out} = \left\lfloor
> \frac{W_{in} + 2 \times padding - dilation \times (K - 1) - 1}{S} + 1
> \right\rfloor
> }
> $$

上上图在Flatten和Fully connected 中间少了个展平后的1024（$64 * 4 * 4$）,1024与64之间还有全连接层（如黄框所示）,64 与10之间也还有全连接层（如黄框所示）

 ![image-20251029203457518](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104451.png)

```python
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        self.conv1 = Conv2d(3, 32, 5, padding=2)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = Conv2d(32, 32, 5, padding=2)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = Conv2d(32, 64, 5, padding=2)
        self.maxpool3 = MaxPool2d(2)
        self.flatten = Flatten()
        self.Linear1 = Linear(1024, 64)
        self.Linear2 = Linear(64, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x
     
tudui = Tudui()
print(tudui)   
# 写完检查网络正确性，主要去看输出的值是否符合要求
input = torch.ones(64, 3, 32, 32)
# 表示由64张图
output = tudui(input)
print(output.shape)
# torch.Size([64, 10])
```

![image-20251029203959035](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104422.png)

```python
# 引入Sequential
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        
        self.modle1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        		
        )
    
    def forward(self, x):
        x = self.model1(x)
        return x
     
tudui = Tudui()
print(tudui)   
# 写完检查网络正确性，主要去看输出的值是否符合要求
input = torch.ones(64, 3, 32, 32)
# 表示由64张图
output = tudui(input)
print(output.shape)
# torch.Size([64, 10])
```

引入Sequential,代码便简洁很多

```python
# 引入tensorboard可视化
from torch import nn
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, Flatten, Linear
from torchvision import SummaryWriter

class Tudui(nn.Module):
    def __init__(self):
        super(Tudui, self).__init__()
        
        self.modle1 = Sequential(
            Conv2d(3, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 32, 5, padding=2),
            MaxPool2d(2),
            Conv2d(32, 64, 5, padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024, 64),
            Linear(64, 10)
        		
        )
    
    def forward(self, x):
        x = self.model1(x)
        return x
     
tudui = Tudui()
print(tudui)   
# 写完检查网络正确性，主要去看输出的值是否符合要求
input = torch.ones(64, 3, 32, 32)
# 表示由64张图
output = tudui(input)
print(output.shape)
# torch.Size([64, 10])

writer = SummaryWriter("../logs_seq")
writer.add_graph(tudui, input)# 绘制计算图
writer.close()
```

在terminal 中输入tensorboard --logdir=logs_seq,点击输出的端口，在弹出的网页中可以看到输出结果

![image-20251029205833017](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104350.png)

![image-20251029205855738](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104843.png)

![image-20251029205915263](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104601.png)

继续点击Linear[7]

![image-20251029205949853](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104610.png)

黄色框会显示送到网络中数据尺寸的大小

![image-20251029210010286](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062104619.png)