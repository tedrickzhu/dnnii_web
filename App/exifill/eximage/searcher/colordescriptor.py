#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:32
# @Author  : zhuzhengyi
# @Desc    : 颜色空间特征提取器
# @File    : colordescriptor.py
# @Software: PyCharm

import cv2
import numpy

class ColorDescriptor:
	__slot__ = ["bins"]

	def __init__(self, bins):
		self.bins = bins

	# 得到图片的色彩直方图，mask为图像处理区域的掩模
	def getHistogram(self, image, mask, isCenter):
		# 利用OpenCV中的calcHist得到图片的直方图
		imageHistogram = cv2.calcHist([image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256])
		# 标准化(归一化)直方图normalize
		imageHistogram = cv2.normalize(imageHistogram, imageHistogram).flatten()
		# isCenter判断是否为中间点，对色彩特征向量进行加权处理
		if isCenter:
			weight = 5.0  # 权重记为0.5
			for index in range(len(imageHistogram)):
				imageHistogram[index] *= weight
		return imageHistogram

	# 将图像从BGR色彩空间转换为HSV色彩空间
	def describe(self, image):
		image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
		features = []
		# 获取图片的中心点和图片的大小
		height, width = image.shape[0], image.shape[1]
		centerX, centerY = int(width * 0.5), int(height * 0.5)
		# initialize mask dimension
		# 生成左上、右上、左下、右下、中心部分的掩模。
		# 中心部分掩模的形状为椭圆形。这样能够有效区分中心部分和边缘部分，从而在getHistogram()方法中对不同部位的色彩特征做加权处理。
		segments = [(0, centerX, 0, centerY), (0, centerX, centerY, height), (centerX, width, 0, centerY),
		            (centerX, width, centerY, height)]
		# 初始化中心部分
		axesX, axesY = int(width * 0.75) / 2, int(height * 0.75) / 2
		ellipseMask = numpy.zeros([height, width], dtype="uint8")
		cv2.ellipse(ellipseMask, (int(centerX), int(centerY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)
		# cv2.ellipse(ellipMask, (int(cX), int(cY)), (int(axesX), int(axesY)), 0, 0, 360, 255, -1)
		# 初始化边缘部分
		for startX, endX, startY, endY in segments:
			cornerMask = numpy.zeros([height, width], dtype="uint8")
			cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
			cornerMask = cv2.subtract(cornerMask, ellipseMask)
			# 得到边缘部分的直方图
			imageHistogram = self.getHistogram(image, cornerMask, False)
			features.append(imageHistogram)
		# 得到中心部分的椭圆直方图
		imageHistogram = self.getHistogram(image, ellipseMask, True)
		features.append(imageHistogram)
		# 得到最终的特征值
		return features
