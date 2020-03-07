# -*- coding: utf-8 -*-
# @Time    : 20-2-7 下午2:52
# @Author  : zhuzhengyi
import os

basedir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# gmcnn use gpu parameter config
# os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#     "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]))
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Config:
	SECRET_KEY = os.environ.get('SECRET_KEY') or 'mLZXDOv12YwdM9ZG'
	DEBUG = False
	TESTING = False
	# HOSTNAME = '0.0.0.0'
	# PORT = '5555'

class DevelopConfig(Config):
	DEBUG = True
	# SERVER_NAME='0.0.0.0:5555'


