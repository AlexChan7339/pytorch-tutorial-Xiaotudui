# 1-Python学习中的两大法宝函数

package可以看成一个大的工具箱（一开始时关闭的），里面有4层（其中第3层放了a、b、c三个工具）可以放工具的，下面两个函数可以探索工具箱的结构

- `dir()`：打开操作，能让我们知道工具箱以及工具箱的分割取有什么东西

- `help()`：类似说明书，让我们知道每个工具时如何使用的，工具的使用方法

  ![在这里插入图片描述](https://typora3.oss-cn-shanghai.aliyuncs.com/202510191545587.png)

```python
# example1
dir(pytorch)
# out:1、2、3、4

dir(pytorch.3)
# out:a,b,c

help(pytorch.3.a)
# out: 将此板手放在特定地方，然后拧动

# 查看torch包中有哪些分隔区
dir(pytorch)

# 查看torch.cuda.is_available的分隔区
dir(torch.cuda.is_available)
# 发现输出为
# ['__annotations__', '__call__', '__class__', '__closure__', '__code__', '__defaults__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__get__', '__getattribute__', '__globals__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__kwdefaults__', '__le__', '__lt__', '__module__', '__name__', '__ne__', '__new__', '__qualname__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__']
# __XXX__:表示为函数(前后的下划线为规范，表示不能修改，一个函数代表一个道具)
# 查看函数功能
help(torch.cuda.is_available)
# 方便记忆当我需要求助时（被抢劫），身上东西会减少，is_available后面就不需要加（）
```
