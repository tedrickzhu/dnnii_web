#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 19-11-29 下午2:36
# @Author  : zhuzhengyi
# @File    : searchengine.py
# @Software: PyCharm
import multiprocessing
from os.path import join
from App.exifill.eximage.searcher import structuredescriptor, searcher, colordescriptor
import cv2
import glob

'''
单线程太慢，使用多进程，cpython的多线程threading受解释器GIL锁的影响，并不能真正实现多核并发的效率
'''

def buildimgpairs(dataset,colorIndexPath,structureIndexPath,imgfilepath,eximgfilepath,fdbdir=None,coresnums=None):
	if fdbdir is None:
		fdbdir = dataset
	filelist = glob.glob(dataset + "*.png")
	totalnums = len(filelist)

	cores = multiprocessing.cpu_count()
	if (coresnums is None) or (coresnums>cores) :
		pool = multiprocessing.Pool(processes=cores)
	else:
		pool = multiprocessing.Pool(processes=coresnums)

	parameterlist = []
	for i in range(totalnums):
		parameterlist.append([filelist[i],colorIndexPath,structureIndexPath])

	pairlist = pool.map(get_img_pair,parameterlist)

	with open('./imgandeximg.txt','a+') as imgandeximg:
		with open(imgfilepath, "a+") as imgfile:
			with open(eximgfilepath,'a+') as eximgfile:
			# # 匹配结果存储
				for imgname,pairname in pairlist:
					imagePath = join(dataset,imgname+'.png')
					pairpath = join(fdbdir,pairname+'.png')

					# 将匹配结果写入到csv文件中去
					imgfile.write(str(imagePath))
					imgfile.write('\r\n')

					eximgfile.write(str(pairpath))
					eximgfile.write('\r\n')

					imgandeximg.write(str(imagePath)+','+str(pairpath))
					imgandeximg.write('\r\n')

				eximgfile.close()
		# close index file
		imgfile.close()

	imgandeximg.close()
	print('pair img and eximg DONE======', totalnums)


def get_img_pair(parameters):
	imagePath, colorIndexPath, structureIndexPath = parameters[0],parameters[1],parameters[2]
	imageName = imagePath.split("/")[-1].split(".")[0]
	imageSearcher = searcher.Searcher(colorIndexPath, structureIndexPath)
	queryFeatures = None
	queryStructures = None

	idealBins = (8, 12, 3)
	idealDimension = (16, 16)

	# 传入色彩空间的bins
	colorDescriptor = colordescriptor.ColorDescriptor(idealBins)
	# 传入构图空间的bins
	structureDescriptor = structuredescriptor.StructureDescriptor(idealDimension)
	queryImage = cv2.imread(imagePath)
	queryFeatures = colorDescriptor.describe(queryImage)
	queryStructures = structureDescriptor.describe(queryImage)
	###########################################
	# 检索
	searchResults = imageSearcher.search(queryFeatures, queryStructures)
	respairs = []
	for resimgname, score in searchResults:
		if resimgname != imageName:
			respairs.append(resimgname)
	if len(respairs)>0:
		return [imageName,respairs[0]]
	else:
		return [imageName,imageName]


if __name__ == '__main__':
	# dataset = '/home/zzy/work/exi_inpaint/imgs/places2_512x680/'
	dataset = '/home/zzy/TrainData/MITPlace2Dataset/base1000recut/'
	# dataset = '/home/zzy/work/experiment_image_inpaint/searcher/examples/output/'
	# dataset = "/home/zzy/TrainData/HolidaySet/"
	colorIndexPath = "/home/zzy/work/exi_inpaint/eximage/files/b1000colorfeatures.csv"
	structureIndexPath = "/home/zzy/work/exi_inpaint/eximage/files/b1000strucfeatures.csv"
	imgfile = "/home/zzy/work/exi_inpaint/eximage/files/b1000imgfile.txt"
	eximgfile = "/home/zzy/work/exi_inpaint/eximage/files/b1000eximgfile.txt"
	buildimgpairs(dataset, colorIndexPath, structureIndexPath, imgfile,eximgfile)
	print('build image pairs done.')