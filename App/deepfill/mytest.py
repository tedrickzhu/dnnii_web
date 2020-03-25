#-*- coding: utf-8 -*-
# @Time    : 20-2-12 上午10:46
# @Author  : zhuzhengyi

import cv2
import numpy as np
import tensorflow as tf
import neuralgym as ng

from os.path import join

from App.deepfill.inpaint_model import InpaintCAModel
from App.basedata import BaseData

'''
generative_inpainting 说明：
1，places2的模型，github的介绍说的是使用256x256训练的，但是，测试发现，512x680，256x256的图片均可正常使用
2，celebahq_256,测试得，只可用于256x256的celeba或者celebahq图片，celebahq的效果稍好
3,输入的图片可以是原图，也可以是已经画好待修复区域的图，重要的是mask图片，mask图片中包含有选中区域

'''

def deepfill_inpaint(basedata,imagepath,maskinfo,status,checkpointdir,inputimgpath,outputpath):
	# basedata = BaseData()
	FLAGS = ng.Config(join(basedata.DEEPFILL_BASE_DIR,'inpaint.yml'))
	# FLAGS = ng.Config('./inpaint.yml')
	# FLAGS = ng.Config('/home/zzy/work/dnnii_web/dnnii_web/App/deepfill/inpaint.yml')
	# ng.get_gpus(1)
	# args, unknown = parser.parse_known_args()

	model = InpaintCAModel()
	image = cv2.imread(imagepath)
	h, w, _ = image.shape
	if status==0:
	    mask = np.zeros((h, w, 3)).astype(np.uint8)
	    for rect in maskinfo:
	        mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2], :] = 255
	else:
	    mask = cv2.imread(maskinfo)

	mask = cv2.resize(mask, (w,h), fx=0.5, fy=0.5)
	assert image.shape == mask.shape

	#把原始图片划分成grid*grid个格子区域，'//'表示向下取整的除法
	grid = 8
	image = image[:h//grid*grid, :w//grid*grid, :]
	mask = mask[:h//grid*grid, :w//grid*grid, :]
	print('Shape of image: {}'.format(image.shape))

	inputimage = image * ((255 - mask)//255) + mask
	cv2.imwrite(inputimgpath, inputimage.astype(np.uint8))

	image = np.expand_dims(image, 0)
	mask = np.expand_dims(mask, 0)
	input_image = np.concatenate([image, mask], axis=2)

	sess_config = tf.ConfigProto()
	sess_config.gpu_options.allow_growth = True
	deepfill_graph = tf.Graph()
	with tf.Session(config=sess_config,graph=deepfill_graph) as deepfill_sess:
	    input_image = tf.constant(input_image, dtype=tf.float32)
	    output = model.build_server_graph(FLAGS, input_image)
	    output = (output + 1.) * 127.5
	    output = tf.reverse(output, [-1])
	    output = tf.saturate_cast(output, tf.uint8)
	    # load pretrained model
	    vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
	    assign_ops = []
	    for var in vars_list:
	        vname = var.name
	        from_name = vname
	        var_value = tf.contrib.framework.load_variable(checkpointdir, from_name)
	        assign_ops.append(tf.assign(var, var_value))
	    deepfill_sess.run(assign_ops)
	    print('deepfill Model loaded.')
	    result = deepfill_sess.run(output)
	    cv2.imwrite(outputpath, result[0][:, :, ::-1])
	    deepfill_sess.close()

def ceshi():
    pretrained_models={
        "celebahq":"../App/checkpoints/deepfill/celebahq_256",
        "places2":"../App/checkpoints/deepfill/places2_256"
    }
    checkpointdir=pretrained_models["celebahq"]
    inputimgpath = "./tmp_deepfill/input000.png"
    outputpath="./tmp_deepfill/ceshioutput8.png"
    imagepath = "../App/static/images/celeba_256x256/001.png"
    maskpath="../App/static/images/maskimg/center_mask_256.png"
    # image = cv2.imread(imagepath)
    # mask = cv2.imread(maskpath)
    # print(image.shape,mask.shape)
    deepfill_inpaint(imagepath, maskpath,1, checkpointdir, inputimgpath,outputpath)

    print("this is done")

if __name__ == '__main__':
    ceshi()