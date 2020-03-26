import os
import time

import cv2
import random
import numpy as np
import torch
from shutil import copyfile
from App.edge.src.config import Config
from App.edge.src.edge_connect import EdgeConnect

def edgeinp(inputimg,maskinfo,status,checkpointdir,output):

	if status == 0:
		truthimage = cv2.imread(inputimg)
		h, w, _ = truthimage.shape
		mask = np.zeros((h, w, 3)).astype(np.uint8)
		for rect in maskinfo:
			mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :] = 255
		# maskimgpath = './App/static/upload/restore/'+os.path.basename(inputimg).split('.')[0]+'_mask_'+time.strftime('%Y%m%d%H%M%S')+'.png'
		maskimgpath = './App/static/upload/restore/'+os.path.basename(inputimg).split('.')[0]+'_mask.png'
		cv2.imwrite(maskimgpath,mask)

	elif os.path.isfile(maskinfo):
		maskimgpath = maskinfo
	else:
		return False

	config = load_config(inputimg,maskimgpath,checkpointdir,output)
	# cuda visble devices
	# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)

	# init device
	if torch.cuda.is_available():
	    config.DEVICE = torch.device("cuda")
	    torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
	else:
	    config.DEVICE = torch.device("cpu")

	# set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
	cv2.setNumThreads(0)

	# initialize random seed
	torch.manual_seed(config.SEED)
	torch.cuda.manual_seed_all(config.SEED)
	np.random.seed(config.SEED)
	random.seed(config.SEED)

	# build the model and initialize
	model = EdgeConnect(config)
	model.load()

	# model test
	print('\nstart testing...\n')
	resultpathlist = model.test()
	if len(resultpathlist)>0:
	    return resultpathlist[0]
	else:
	    return False


def load_config(inputimg,maskimg,checkpointdir,output):
	# config_path = os.path.join(EDGEBASEDIR, 'config.yml')
	config_path = os.path.join(checkpointdir, 'config.yml')

	# create checkpoints path if does't exist
	if not os.path.exists(checkpointdir):
	    os.makedirs(checkpointdir)
	# copy config template if does't exist
	if not os.path.exists(config_path):
	    copyfile('./App/edge/config.yml.example', config_path)

	# load config file
	config = Config(config_path)
	# test mode
	config.MODE = 2
	config.MODEL = 3
	config.INPUT_SIZE = 0

	config.TEST_FLIST = inputimg
	config.TEST_MASK_FLIST = maskimg
	# config.TEST_EDGE_FLIST = args.edge
	config.RESULTS = output

	return config

if __name__ == '__main__':
    checkpointdir ='/home/zzy/work/dnnii_web/dnnii_web/App/checkpoints/edge/places2/'
    inputimg='./examples/myplaces2/images/001.png'
    maskimg='./examples/myplaces2/masks/001.png'
    output='./examples/results/'
    reslutpaths = edgeinp(inputimg,maskimg,checkpointdir,output)
    if reslutpaths:
	    print(reslutpaths)