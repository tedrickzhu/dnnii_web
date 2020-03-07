# -*- coding: utf-8 -*-
# @Time    : 20-3-3 下午10:19
# @Author  : zhuzhengyi

import cv2
import os
import numpy as np
'''
this method is only use for MIT place2 dataset val_large sub-dataset
you can change it use glob to make it as a common method
#imgsize = (w,h)
'''
def recutimage(basepath,baserecutpath,readnums=0,imgsize=(256,256)):
	if readnums==0:
		readnums = 10000000
	if not os.path.exists(baserecutpath):
		os.mkdir(baserecutpath)
	precount=0
	writeindex = 1
	for readindex in range(1,readnums+1):
		# if readindex != 527:
		# 	continue
		# if writeindex >imgnums:
		# 	break
		filename = 'Places365_val_'+str(readindex).zfill(8)+'.jpg'
		filepath = os.path.join(basepath,filename)
		if not os.path.isfile(filepath):
			print("break,no such file ,",filepath)
			print('readindx:', readindex)
			print('writeindex:', writeindex)
			break
		# print(filepath)
		image = cv2.imread(filepath)
		h,w,c = image.shape
		rate = (imgsize[1]*1.0)/(imgsize[0]*1.0)
		if (h*1.0)/(w*1.0) > rate:
			h_new = int(w*rate)

			image_new = image[:h_new,:,:]
			image_new = cv2.resize(image_new, imgsize)
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath,image_new)
			writeindex+=1

			image_new = image[h-h_new:,:,:]
			image_new = cv2.resize(image_new, imgsize)
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath, image_new)
			writeindex += 1
			#if compare area less than 0.7,cut two new images in the middle of source image
			if float(h-h_new)/float(h) > 0.3:
				h_new_s = int(h_new*0.3)
				image_new = image[h_new_s:(h_new_s+h_new), :, :]
				image_new = cv2.resize(image_new, imgsize)
				filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
				filepath = os.path.join(baserecutpath, filename)
				cv2.imwrite(filepath, image_new)
				writeindex += 1
				h_new_s = int(h-h_new * 1.3)
				image_new = image[h_new_s:(h_new_s + h_new), :, :]
				image_new = cv2.resize(image_new, imgsize)
				filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
				filepath = os.path.join(baserecutpath, filename)
				cv2.imwrite(filepath, image_new)
				writeindex += 1

		elif (h*1.0)/(w*1.0) < rate:
			w_new = int(h / rate)

			image_new = image[:, :w_new, :]
			image_new = cv2.resize(image_new,imgsize)
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath, image_new)
			writeindex += 1

			image_new = image[:, w - w_new:, :]
			image_new = cv2.resize(image_new,imgsize)
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath, image_new)
			writeindex += 1
			if float(w-w_new)/float(w) > 0.3:
				w_new_s = int(w*0.3)
				image_new = image[ :,w_new_s:(w_new_s+w_new), :]
				image_new = cv2.resize(image_new, imgsize)
				filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
				filepath = os.path.join(baserecutpath, filename)
				cv2.imwrite(filepath, image_new)
				writeindex += 1
				w_new_s = int(w-w_new * 1.3)
				image_new = image[:, w_new_s:(w_new_s + w_new), :]
				image_new = cv2.resize(image_new, imgsize)
				filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
				filepath = os.path.join(baserecutpath, filename)
				cv2.imwrite(filepath, image_new)
				writeindex += 1

		elif (h*1.0)/(w*1.0) == rate:
			image_new = cv2.resize(image,imgsize)
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath, image_new)
			writeindex += 1
			filename = 'Places365_' + str(writeindex).zfill(8) + '.png'
			filepath = os.path.join(baserecutpath, filename)
			cv2.imwrite(filepath, image_new)
			writeindex += 1

		if writeindex-precount>500:
			precount=writeindex
			print('readindex:',readindex)
			print('writeindex:',writeindex)
	print('======recut images finished======')
	print('have read source images numbers:', readindex)
	print('have written recutted image numbers:', writeindex)

def create_mask(imgshap,whitesize,outputpath):
	h,w = imgshap[0],imgshap[1]
	mask = np.zeros((h, w, 3)).astype(np.uint8)
	h_s = int((h-whitesize)/2)
	w_s = int((w-whitesize)/2)
	mask[h_s:h_s + whitesize, w_s:w_s + whitesize, :] = 255
	cv2.imwrite(outputpath,mask)


if __name__ == '__main__':
	basepath = '/home/zzy/TrainData/MITPlace2Dataset/val_large'
	baserecutpath = '/home/zzy/TrainData/MITPlace2Dataset/base1000recut/'

	# imgsize=(256,256)
	recutimage(basepath,baserecutpath,1000)
	# imgshap = (512,680)
	# for whitesize in [100,128,200]:
	# 	outputpath = './center_mask_512x680_'+str(whitesize)+'.png'
	# 	create_mask(imgshap,whitesize,outputpath)