#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:34
# @Author  : zhuzhengyi
# @Desc    : 构图空间提取器
# @File    : structuredescriptor.py
# @Software: PyCharm
import cv2
# 将图片进行归一化处理，返回HSV色彩空间矩阵
class StructureDescriptor:
    __slot__ = ["dimension"]
    def __init__(self, dimension):
        self.dimension = dimension
    def describe(self, image):
        image = cv2.resize(image, self.dimension, interpolation=cv2.INTER_CUBIC)
        # 将图片转化为BGR图片转化为HSV格式
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # print(image)
        return image