# -*- coding: utf-8 -*-
# @Time    : 20-2-22 下午11:43
# @Author  : zhuzhengyi

# import os
import subprocess
# import numpy as np

# gmcnn use gpu parameter config
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#     "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

for x in subprocess.Popen("nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines():
	print(x)
	print(x.split())