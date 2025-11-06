# 4-Dataset实战

将数据集文件夹放在pycharm的环境下面，文件夹名字修改为`dataset`

```python
# 可以现在python Console(交互式控制台)进行调试
from PIL import Image
img_path =  "C:\\Users\\sejje\\PycharmProjects\\xiaotudui-project\\dataset\\hymenoptera_data\\hymenoptera_data\\train\\ants\\0013035.jpg"
# 需要用\\ 对\进行转义

# 读取图片
img = Image.open(img_path)
# 此时img变量的属性为JpegImageFile

# 查看图片大小
img.size
# (768, 512)：宽为768，高为512

# 显示图片
img.show()

# 获取文件夹下面所有文件，用list封装
dir_path = "./dataset/hymenoptera_data/hymenoptera_data/train/ants"# 相对路径不需要转义字符
import os
img_path_list = os.listdir(dir_path)

```

![image-20251019173829439](https://typora3.oss-cn-shanghai.aliyuncs.com/202511062110867.png)

<center><font color=RED>文件夹下面所有图片</font></center>



```python
# 利用ants 和bees文件夹改写getitem（获取img和target）和len方法
from torch.utils.data import Dataset
from PIL import Image	# 读取图像
import os # 关于系统的库

class MyDaya(Dataset):
    # 继承在Dataset类下面
    def __init__(self, root_dir, label_dir):
        # 初始化类：根据Dataset类去初始化特列实例，为整个class提供全局变量，为getitem和len方法提供量， 可以放在后面写
        self.root_dir = root_dir
        # self:指定一个类中的全局变量，是的函数中的变量可以在另一个函数中使用
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 将两个路径进行拼接，中间用\\连接（windows中），比如"dataset/train\\ants"
        self.img_path = os.list_dir(self.path)# 获取路径下所有地址
        
    def __getitem__(self, idx):
        # 根据索性获取图像（需要所有图片位置的list）和其地址、label值
        img_name = self_img_path[idx]# 引用全局变量
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)# 图片路径
        img = Imgae.open(img_item_path)
        label = self.label_dir
        return img, label
    
    def __len__(self):
        return len(self.img_path)
    
root_dir = "/dataset/hymenoptera_data/hymenoptera_data/train"# 相对路径
ants_label_dir = "ants" 
ants_dataset = MyData(root_dir, ants_label_dir)# 初始化实例
img, label = ants_dataset[0]
# 返回结果为image格式和label值
img.show() # 显示图片

# 获取所有图像（蚂蚁和蜜蜂）
bees_label_dir = "bees"
bees_dataset = MyData(root_dir, bees_label_dir)# 初始化实例
train_dataset = ants_dataset + bees_dataset
# 两个数据集叠加在一起（在axis=0方向上）
# 作用：数据集不足时需要额外数据集进行扩充，或者获取子数据集

# 检验
len(train_dataset)
len(bee_trainset)
img, label = train_dataset[124]
```
