# -*- coding: utf-8 -*-
# @Time    : 20-3-4 下午10:29
# @Author  : zhuzhengyi

import scipy.io
import numpy as np
import os
import scipy.misc

cwd = os.getcwd()
VGG_PATH = cwd + "/imagenet-vgg-verydeep-19.mat"
vgg = scipy.io.loadmat(VGG_PATH)
# 先显示一下数据类型，发现是dict
print(type(vgg))
# 字典就可以打印出键值dict_keys(['__header__', '__version__', '__globals__', 'layers', 'classes', 'normalization'])
print(vgg.keys())
# 进入layers字段，我们要的权重和偏置参数应该就在这个字段下
layers = vgg['layers']

# 打印下layers发现输出一大堆括号，好复杂的样子：[[ array([[ (array([[ array([[[[ ,顶级array有两个[[
# 所以顶层是两维,每一个维数的元素是array,array内部还有维数
# print(layers)

# 输出一下大小，发现是(1, 43)，说明虽然有两维,但是第一维是”虚的”,也就是只有一个元素
# 根据模型可以知道,这43个元素其实就是对应模型的43层信息(conv1_1,relu,conv1_2…),Vgg-19没有包含Relu和Pool,那么看一层就足以,
# 而且我们现在得到了一个有用的index,那就是layer,layers[layer]
print("layers.shape:", layers.shape,type(layers))
layer = layers[0]
# 输出的尾部有dtype=[('weights', 'O'), ('pad', 'O'), ('type', 'O'), ('name', 'O'), ('stride', 'O')])
# 可以看出顶层的array有5个元素,分别是weight(含有bias), pad(填充元素,无用), type, name, stride信息,
# 然后继续看一下shape信息,
print("layer.shape:", layer.shape,type(layer))
# print(layer)输出是(1, 1),只有一个元素
print("layer[0].shape:", layer[0].shape,type(layer[0]))
# layer[0][0].shape: (1,),说明只有一个元素
print("layer[0][0].shape:", layer[0][0].shape,type(layer[0][0]))

# layer[0][0][0].shape: (1,),说明只有一个元素
print("layer[0][0][0].shape:", layer[0][0][0].shape,type(layer[0][0][0]))
# len(layer[0][0]):5，即weight(含有bias), pad(填充元素,无用), type, name, stride信息
print("len(layer[0][0][0]):", len(layer[0][0][0]))
# 所以应该能按照如下方式拿到信息，比如说name，输出为['conv1_1']
print("name:", layer[0][0][0][3])
# 查看一下weights的权重，输出(1,2),再次说明第一维是虚的,weights中包含了weight和bias
print("layer[0][0][0][0].shape", layer[0][0][0][0].shape,type(layer[0][0][0][0]))
print("layer[0][0][0][0].len", len(layer[0][0][0][0]))

# weights[0].shape: (2,),weights[0].len: 2说明两个元素就是weight和bias
print("layer[0][0][0][0][0].shape:", layer[0][0][0][0][0].shape,type(layer[0][0][0][0][0]))
print("layer[0][0][0][0].len:", len(layer[0][0][0][0][0]))

weights = layer[0][0][0][0][0]
# 解析出weight和bias
weight, bias = weights
# weight.shape: (3, 3, 3, 64)
print("weight.shape:", weight.shape)
# bias.shape: (1, 64)
print("bias.shape:", bias.shape)

# python train.py --dataset places2 --data_file /home/zhengyi_zhu/work/exi_inpaint/places2_512x680_file.txt