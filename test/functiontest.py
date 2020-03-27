# -*- coding: utf-8 -*-
# @Time    : 20-2-15 下午9:55
# @Author  : zhuzhengyi

# def multiarg(*arg):
# 	print(arg[0],arg[1])
#
# x=12
# y=56
# multiarg(x,y)
# # import time
# a = time.strftime('%Y%m%d%H%M%S')
# print(type(a),a)

# imagepath = './App/static/images/gmcnn_256x256/0023.png'
#
# a = imagepath.split("static/")
# print(a)

# rectmasks = [1,2,3,4,5,6,7,8,9,6,7,4,3,2,5,9]
# masklocs = []
# print(len(rectmasks))
# for num in range(0,len(rectmasks)-3,4):
# 	onerect = []
# 	onerect.append(rectmasks[num])
# 	onerect.append(rectmasks[num+1])
# 	onerect.append(rectmasks[num+2])
# 	onerect.append(rectmasks[num+3])
# 	masklocs.append(onerect)
# 	print(onerect,masklocs)
# print(masklocs)

# import os,cv2
#
# imgsdir = '/home/zzy/dnnii_web/App/static/images/celebahq_256x256'
#
# filelist = os.listdir(imgsdir)
# for filename in filelist:
# 	imgpth = os.path.join(imgsdir,filename)
# 	respth = os.path.join(imgsdir,filename.split('.')[0]+'.png')
# 	image = cv2.imread(imgpth)
# 	# image = cv2.resize(image,(256,256))
# 	cv2.imwrite(respth,image)

from multiprocessing import Process

def train(a,b):
	print('asdfasdfasdf')
	print(a,type(a),b,type(b))

if __name__=='__main__': # 在windows必须在这句话下面开启多进程
	# p = Process(target=train)
	# p.start()
	# p.join() # 进程结束后，GPU显存会自动释放
	kwargs = {'a':'asdf','b':'qwer'}
	p = Process(target=train,kwargs=kwargs) #新训练
	p.start()
	p.join()
