# 5-Tensor board的使用

> 主要用于查看训练过程中的loss变化，从而知道训练过程中是否是正常；也可以从loss中从多少轮（下图的`step 2974`）后符合我们的预期。

![image-20251019202003322](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111140.png)

## 5.1-SummaryWriter的使用

```python
# 导入类
from torch.utils.tensorboard import SummaryWriter

# 查看类的说明文档(也可以按下Ctrl，点击上一行代码中的SummaryWriter)
help(SummaryWriter)
```

- 它可以直接向`log_dir`写入文件，并被TensorBoard进行解析
- 初始化函数
  - log_dir: 保存路径，也有默认路径（值为None）
  - comment: 在文件名的后面加后缀（learningRate_batchsize）

![image-20251019202446113](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111843.png)

```python
# 导入类
from torch.utils.tensorboard import SummaryWriter

# 创建实例
writer = SummaryWriter("logs")

# 添加图片、数
writer.add_image()
writer.add_scalar()

writer.close()
```

构建类似效果

![image-20251019202836276](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111259.png)

## 5.2-add_scalar()的使用

- 作用：添加标量数据到summary中
- 参数
  - tag(string):图标的标题
  - scalar_value: 保存数值（上图的y轴）
  - global_step：训练到多少轮时对应的训练轮数数值（上图的x轴）

```python
# 导入类
from torch.utils.tensorboard import SummaryWriter

# 创建实例
writer = SummaryWriter("logs")

# 添加标量数据
for i in range(100):
	writer.add_scalar(tag = 'y=x', scaler_value=i, global_step=i)# 绘制y=x的图像

writer.close()
```

运行上面需要保证TensorBoard已经安装

> 在特定虚拟环境下执行
>
> ```python
> pip install tensorboard
> ```

运行完成的log文件会放在该项目下的logs的文件夹里

## 5.3-打开log文件

### 5.3.1-法一：命令行

在特定虚拟环境下，运行(logdir=事件文件所在文件夹名)，运行后直接点击6006端口，有可能会很多人都在使用这个端口，可以指定端口

```python
tensorboard --logdir=logs
```

![image-20251019203924195](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111706.png)

指定端口,避免和其他人一样

```python
tensorboard --logdir=logs --port=6007
```

运行结果（log），将鼠标放在线上可以看到具体的数值（比如train_loss, val_loss）

![image-20251019204130164](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111646.png)

```python
# 拟合直线计算方式
def smooth(scalars: list[float], weight: float) -> list[float]:
    """
    EMA implementation according to
    https://github.com/tensorflow/tensorboard/blob/34877f15153e1a2087316b9952c931807a122aa7/tensorboard/components/vz_line_chart2/line-chart.ts#L699
    """
    last = 0
    smoothed = []
    num_acc = 0
    for next_val in scalars:
        last = last * weight + (1 - weight) * next_val
        num_acc += 1
        # de-bias
        debias_weight = 1
        if weight != 1:
            debias_weight = 1 - math.pow(weight, num_acc)
        smoothed_val = last / debias_weight
        smoothed.append(smoothed_val)
    return smoothed
```

> ### 1) EMA 的递推式（常见写法）
>
> 设衰减率为 $ \beta\in[0,1)$，输入序列为 $x_1,x_2,\dots$。常见的指数移动平均（EMA）递推是
> $$
> m_t = \beta\, m_{t-1} + (1-\beta)\, x_t,
> $$
> 并且通常把初始值 $m_0$ 设为 0（这是算法实现上的常用做法，简单但会引入偏差）。
>
> 这里 $(1-\beta)$ 可看作“当前样本的权重”，而过去样本 $x_{t-k}$ 的权重是 $(1-\beta)\beta^{k}$。
>
> ------
>
> ### 2) 展开可见的加权和形式
>
> 把上面的递推式展开（以 $m_0=0$ 为例），得到
> $$
> m_t = (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i}\, x_i \;+\; \beta^{t} m_0.
> $$
> 因为 $m_0=0$，第二项为 0，所以
> $$
> \boxed{m_t = (1-\beta)\sum_{i=1}^{t} \beta^{\,t-i}\, x_i.}
> $$
> 这清楚地表明 $m_t$ 是历史样本的加权和，但**这些权重之和不一定等于 1**（除非 $t$ 很大使得 $\beta^t$ 接近 0）。
>
> ------
>
> ### 3) 权重之和（就是你说的 debias_weight）
>
> 计算这些权重的和：
> $$
> \sum_{i=1}^{t} (1-\beta)\beta^{\,t-i} = (1-\beta)\sum_{k=0}^{t-1}\beta^{k} = (1-\beta)\frac{1-\beta^{t}}{1-\beta} = 1-\beta^{t}.
> $$
> 所以权重之和恰好是 $1-\beta^{t}$。这个量就是常说的 **debias_weight**。
>
> ------
>
> ### 4) 为什么要除以 debias_weight（即为什么存在“偏差”）
>
> 因为我们通常想把 EMA 解释为“一个加权平均”（weights 应该和为 1）。但由于初始化 $m_0=0$，当前得到的 $m_t$ 的权重之和是 $1-\beta^{t}$（< 1），因此 $m_t$ 实际上比“理想的”加权平均偏小（数值被整体压低了），特别是在训练早期（$t$ 小）时偏差很明显。
>
> 为了把它变成真正规格的加权平均，应当把 $m_t$ 除以权重之和：
> $$
> \hat m_t \;=\; \frac{m_t}{1-\beta^{t}} \;=\; \frac{(1-\beta)\sum_{i=1}^{t}\beta^{\,t-i} x_i}{1-\beta^{t}}
> $$
> 于是 $\hat m_t$ 的权重之和就是 1（无偏的加权平均）。

**注意**：

如果tag和scalar_value不一致时，需要将log文件删掉，再将过程删掉（Ctrl+C）

```python
# 导入类
from torch.utils.tensorboard import SummaryWriter

# 创建实例
writer = SummaryWriter("logs")

# 添加标量数据
for i in range(100):
	writer.add_scalar(tag = 'y=2x', scalar_value=3*i, global_step=i)

writer.close()
```

![image-20251019224815174](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111642.png)

中间那条曲线是如下计算的，相当于scalar_value = [2 * i for i in range(100)] + [3 * i for i in range(100)], 两个列表叠加在一起计算的，而（0， 98 ）则是根据由一开始y=2x 转换到（0， 98）的直线，98是根据指数平滑法计算得到的，0则是y=3x 横坐标其实是从0开始，相当于是拟合后的直线

## 5.4-add_image()的使用

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")

writer.add_image()

for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
    
writer.close()
```

按住`Ctrl`点击add_image,查看说明文档：

![image-20251021222859523](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111949.png)

- tag:图表标题
- img_tensor:tensor、numpy、string等图像类型类型

```python
image_path= "data/train/ants_image/0013035.jpg"

from PIL import Image

img = Image.open(image_path)
print(type(img))
# 'PIL. JpegImagePlugin. JpegImageFile'>
# 该类型不属于 tensor、string、numpy类型，需要对其进行转换
# 下节课会讲如何转化为tensor类型
```

### 利用Opencv读取图片，获取numpy类型图片数据

```python
# 要在指定环境下安装
pip install opencv-python
```

```python
# 在交互台进行调试
import numpy as np

img_array = np.array(img)
print(type(img_array)
#  <class 'numpy.ndarray'>,符合 img_tensor类型
```

```python
# 修改上节代码
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path= "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)

writer.add_image(tag="test", img_tensor=img_array, global_step=1)

for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
    
writer.close()
# 会报错，bug出在img_array上，再次按下Ctrl查看add_image说明文档
```

![image-20251021224327880](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111211.png)

img_tensor的shape默认类型为(3, H, W),其中3表示3通道；而（H, W, 3）也可以，需要设置参数dataformats，也就是修改C

> 说明文档的案例![image-20251021224832144](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111196.png)

```python
# 在交互敞口进行
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path= "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(iamge_path)
img_array = np.array(img_PIL)

```

可以看到img_array的shape是（512， 768， 3）,符合说明文档类型

```python
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path= "data/train/ants_image/0013035.jpg"
img_PIL = Image.open(iamge_path)
img_array = np.array(img_PIL)

writer.add_image(tag="test", img_tensor=img_array, global_step=1, dataformats="HWC")

for i in range(100):
    writer.add_scalar("y=2x", 3*i, i)
    
writer.close()
```

点击Terminal端口网站，在网站页面点击刷新，可以看到图像



**作用**

- 在训练过程中可以看到给model提供哪些数据；
- 在预测过程中，每个阶段的测试结果

## 5.5-touchvision中的transforms

- transform主要是对图片进行变换

- 新建一个脚本，命名为`P9_Transforms`

### 5.5.1-transforms的结构

```python
from touchvision import transforms
# 按住Ctrl点击transforms， 再按住Ctrl点击transforms，查看源码
```

他的架构和用法都集成再他的python文件里，可以发现源代码有很多的类，这些类可以看成是transform工具包里的不同的工具，比如说下图的object就是图像

![image-20251021225943216](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111594.png)

点击左边的structure，可以查看源码的结构，在页面里出现很多名称（蓝色图标的C表示class， M表示method）,以Compose(object)类下面的`__init__(self, transforms)`为例，它是将多个transforms 结合在一起

红框表示：一张图片进来后，先经过中心裁剪（CenterCrop），再经过转化为Tensor（ToTensor）

![image-20251023222101299](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111642.png)

另一个用的比较多的是ToTensor类，它是将图片或者numpy数组转化为Tensor

![image-20251023222324619](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111163.png)

其他用的比较多的的`ToPILImage`（转化为图片）、`Normalize`（正则化）、`Resize`(尺寸修改)、`CenterCrop`（中心裁剪），这些都可以看成是transform.py里的工具，用特定格式的图片经过工具处理，进而输出我们想要的图片结果

![image-20251023222803155](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111220.png)

----

### 5.5.2-transforms的用法

tensor的数据类型

> 相较于普通的数据类型有什么区别，主要通过`transforms.ToTensor`去解决两个问题：
>
> 1. transforms改如何使用（python）
> 2. 为什么需要Tensor数据类型

![image-20251023223230480](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111579.png)

上节提到ToTensor类是将图片或者numpy数组转化为Tensor，当我们调用`__call__()`方法时，需要传入pic变量（可以是图像或者numpy数组）

```python
from touchvision import transforms
from PIL import Image

img_path = "data/train/ants_image/0013035.jpg" # 相对路径
img = Image.open(img_path)	# python中PIL库内置的类
print(img)

```

![image-20251023223741216](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111785.png)

- 类型：PIL.Image图片类型
- 模式：RGB类型
- 尺寸：768 * 512
- 0x275820FEF60：逻辑地址

```python
# 解决第一个问题：transforms改如何使用
tensor_trans = transforms.ToTensor()# 创建实例
tensor_img= tensor_trans(img)# 相当于调用ToTensor类中的__call__方法
# 将光标放在括号里，按下Ctrl+P可以看到需要加入的参数（这里是pic）
# 实现将PIL.Image 变成tensor类型
```

![image-20251023224634012](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111926.png)

完善图片：首先创建具体的工具(上面代码用到的则是`transforms.ToTensor()`),然后根据具体的需求使用工具，使用过程中需要有输入（图像）和输出（我们需要的数据）

```python
# 解决第二个问题：为什么需要tensor数据类型
# 在交互控制台进行（Console）
from touchvision import transforms
from PIL import Image

img_path = "data/train/ants_image/0013035.jpg" 
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img= tensor_trans(img)
# 来查看tensor有哪些参数
```

![image-20251023224939155](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111941.png)

- _backward_hooks:神经网络中的反向传播钩子，利用结果对前面的参数进行调整

- _grad_fn:梯度函数

- data:数据

- device:CPU/GPU

  > tensor包含了神经网络中的理论基础参数（神经网络中肯定需要转换为Tensor类型，在对其进行训练）

也可以用OpenCV进行读取图片

```python
# 打开Terminal，调整到虚拟环境,安装opencv
pip install opencv-python
```

```python
# 在交互控制台进行（Console）
import cv2

img_path = "data/train/ants_image/0013035.jpg" 
cv_img = cv2.imread(img_path)
# 在右边查看cv_img的类型为ndarray,该类型也符合ToTensor的输入类型
```

```python
# 在pycharm代码区域
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

img_path = "data/train/ants_image/0013035.jpg" 
img = Image.open(img_path)
writer = SummaryWriter("logs")

tensor_trans = transforms.ToTensor()
tensor_img= tensor_trans(img)

writer.add_image("Tensor_img", tensor_img)
writer.close()
```

可以看到坐标Project页面的logs文件夹里会生成一个新的log文件，然后在Terminal中输入`tensorboard --logdir=logs`,点击显示的端口，可以在显示的页面中看到图片

## 5.6-常见的Transforms

Transforms主要关注输入、输出和作用（通过不同方法来实现）

![image-20251025162918777](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111988.png)

> 准备一张pytorch图片放在pycharm项目下的images文件夹里
>
> ![下载](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111210.jpeg)
>
> 新建python文件`P10_UsefulTransforms`

```python
from PIL import Image
from torchvision import *

img = Image.open("images/pytorch.png")
print(img)
# 类型为PIL.PngImage，大小为3200*1800
```

## 5.6.1-python中`__call__`的用法

在pycharm项目中新建test文件夹，在文件夹里新建CallTest.py文件

```python
Class Person:
    def __call__(self, name):
        print("__call__" + " Hello "+ name)
        
    def hello(self, name):
        print("hello" +name)
        
person = Person()# 新建Person实例
Person("Zhangsan")# 可以将光标在括号里，按下Ctrl+P键，可以看到参数提示
# __call__ Hello zhangsan
person.hello("lisi")
# hellolisi
```

- `__call__`的调用可以不用`.方法`进行调用，直接对象+括号，并在括号里输入参数

### 5.6.2-python中`ToTensor()`的用法

```python
from touchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
img = Image.open("images/pytorch.png")
print(img)
# 类型为PIL.PngImage，大小为3200*1800

trans_totensor = transforms.ToTensor()
img_tensor = trans_totensor(img)
writer.add_image("ToTensor", img_tensor)
writer.close()
# 运行前，将之前的日志删除，也可以不删除，此时会在之前创建好的logs文件夹下新生成一个log日志
```

可以看到坐标Project页面的logs文件夹里会生成一个新的log文件，然后在Terminal中输入`tensorboard --logdir=logs`,点击显示的端口，可以在显示的页面中看到图片

### 5.6.3-Normalize的使用

用均值或或者标准差参数（n个通道的值）来进行归一化tensor类型的image

![image-20251025170503955](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111708.png)

作用：比如会将像素值的范围从【0，1 】变为【0， 255】

```python
print(img_tensor[0][0][0])
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(img_tensor)fromfro
print(img_tensor[0][0][0])
writer.add_image("Normalize", img_norm)
writer.close()
```

![image-20251025171256780](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111741.png)

<center>色调发生改变</center>



### 5.6.2-python中`Resize()`的用法

- 作用：将输入PIL图像转换成需要的尺寸
  - 如果size是一个序列（h, w），将图片调整为该尺寸
  - 如果size是一个数，将图片等比例进行缩放，图片的最短边会匹配这个值，另一条边则是size * height /wdth (等比例缩放)

![image-20251025171923232](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111722.png)

```python
print(img.size)
# (3200, 1800)
trans_resize = transforms.Resize((512, 512))
img_resize = trans_resize(img)
print(img_resize)
# <PIL. Image. Image image mode=RGB size=512x512 at 0x244830EA860>
# 尺寸变成512 * 512， 类型却是PIL，需要将其转化为Tensor类型
img_resize = trans_totensor(img_resize)
writer.add_image("Resize", img_resize, step=0)
print(img_resize)
writer.close()
```

> pycharm 取消首字母匹配，更好进行代码Tab补充：
>
> 打开Setting页面，搜索栏搜索case，点击General下面的Code Completion，取消勾选Match Case
>
> ![](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111063.png)

图片变得更加窄长

![image-20251025173148792](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111376.png)

### 5.6.3-`Compose()`的用法

```python
trans_resize_2 = transforms.Resize(512)
trans_compose = transforms.Compose([trans_resize_2, trans_totensor])
# 只用不同transofrms对应的输入输出类型（PIL or tensor）
img_reize_2 = trans_compose(img)
writer.add_image("Resize", img_resize, 1)
writer.close()
```

![image-20251025173508201](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111447.png)

![image-20251025173703156](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111744.png)

拉动红框中的橘色拉点，就可以看见原图像经过一步step，图像尺寸进行拉伸，不同的步数可以看到不同的结果，在后面的训练过程中可以看到不同epoch时的训练结果

### 5.6.4-`RandomCrop`的用法

随机裁剪的参数

- size:可选类型为序列（裁剪为【h, w】）或者整数（裁剪为正方形）
- 输入为PIL Image， 输出为PIL Image

```python
trans_random = transforms.RendomCrop(512)
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):# 裁剪10个
    img_crop= trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()
```

```python
trans_random = transforms.RendomCrop((500, 1000))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):# 裁剪10个
    img_crop= trans_compose_2(img)
    writer.add_image("RandomCropHW", img_crop, i)

writer.close()
```

### 5.6.5-总结

1. 关注类的输入和输出类型

   ![image-20251025175138610](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111757.png)

2. 多看官方文档

3. 关注方法需要什么参数

   看一个类时，首先关注`__init__`的参数，对于参数的取值和解释可以通过查看类与`__init__`中间的代码标注类型

   ![image-20251025175409206](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062111908.png)

   如果说明文档没有告诉输出类型时，可以在代码中`print()`来显示，或者使用断点debug