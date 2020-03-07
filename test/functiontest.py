# -*- coding: utf-8 -*-
# @Time    : 20-2-15 下午9:55
# @Author  : zhuzhengyi

def multiarg(*arg):
	print(arg[0],arg[1])

x=12
y=56
multiarg(x,y)
# import time
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