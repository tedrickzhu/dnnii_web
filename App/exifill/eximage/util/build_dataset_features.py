#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:37
# @Author  : zhuzhengyi
# @File    : index.py
# @Software: PyCharm

from searcher.colordescriptor import ColorDescriptor
from searcher.structuredescriptor import StructureDescriptor
import glob
import cv2

def buildfeaturesdb(dataset,colorfeas,strucfeas):
	idealBins = (8, 12, 3)
	colorDesriptor = ColorDescriptor(idealBins)
	filelist = glob.glob(dataset + "*.png")
	count = 0
	totalnums = len(filelist)
	with open(colorfeas, "w") as output:
		# 色彩空间的特征存储
		for imagePath in filelist:
			imageName = imagePath.split("/")[-1].split(".")[0]
			image = cv2.imread(imagePath)
			features = colorDesriptor.describe(image)
			# 将色彩空间的特征写入到csv文件中去
			features = [str(feature).replace("\n", "") for feature in features]
			output.write("%s,%s\n" % (imageName, ",".join(features)))
			count+=1
			if count%500==0:
				print('write color features to csv file:',count,'/',totalnums)
		# close index file
	output.close()
	print('write color features to csv file. DONE===', count, '/', totalnums)
	count = 0

	idealDimension = (16, 16)
	structureDescriptor = StructureDescriptor(idealDimension)

	with open(strucfeas, "w") as output:
		# 构图空间的色彩特征存储
		for imagePath in filelist:
			imageName = imagePath.split("/")[-1].split(".")[0]
			image = cv2.imread(imagePath)
			structures = structureDescriptor.describe(image)
			# 将构图空间的色彩特征写入到文件中去  write structures to file
			structures = [str(structure).replace("\n", "") for structure in structures]
			output.write("%s,%s\n" % (imageName, ",".join(structures)))
			count+=1
			if count%500==0:
				print('write structure features to csv file:',count,'/',totalnums)
		# close index file
	output.close()
	print('write structure features to csv file. DONE===', count, '/', totalnums)

if __name__ == '__main__':
	dataset = '/home/zzy/TrainData/MITPlace2Dataset/base1000recut/'
	# dataset = "/home/zzy/TrainData/HolidaySet/"
	colorfeas = "/home/zzy/work/exi_inpaint/eximage/files/b1000colorfeatures.csv"
	strucfeas = "/home/zzy/work/exi_inpaint/eximage/files/b1000strucfeatures.csv"
	buildfeaturesdb(dataset, colorfeas, strucfeas)
	print('build features file done.')