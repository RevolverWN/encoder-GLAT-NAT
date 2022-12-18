'''

消息传递
不同抽象层次的交互
把它的程序当做自己的用每个模块

抽象：1 隐藏细节 2 泛化 3 概念和实现

抽象有很多不同的层次
the user should not be able to bypass the first layer, the user interface, to look at the codebase, for example.
This concept is known as the abstraction barrier: the layer of abstractions are normally isolated.

generalization is the main purpose of an abstract class

先加载验证集，训练集、模型、损失函数和优化器后定义，要从检查点恢复
'''


# from import的过程中仍然会执行这个被导入的文件，原因是程序执行完才知道导入的变量是否发生改变
# from import可以用导入模块的任何变量


# 1 在函数内部导入模块，不用全局导入