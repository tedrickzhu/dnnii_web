# -*- coding: utf-8 -*-
# @Time    : 20-2-10 下午10:12
# @Author  : zhuzhengyi
import os, time

from App.conenc.mytest import conenc_inp
from App.deepfill.mytest import deepfill_inpaint
from App.edge.mytest import edgeinp
from App.exifill.exigmcnn.exifill_options import ExiFillOptions
from App.exifill.mytest import exifill_inpaint
from App.gmcnn.mytest import gmcnn_inpaint
from App.gmcnn.options.gmcnn_options import GMCNNOptions
from App.conenc.conenc_options import ConEncOptions

'''
每个函数均返回一个字典，格式如下
result={
	'modelname':modelname,
	'inputimgpath':inputimgpath,
	'resultimgpath':resultimgpath,
	'truthimgpath':truthimgpath,
}
'''

def gmcnn_center_inp(basedata,dataset,imagepath):
	if dataset in basedata.PRE_MODEL_DIR['gmcnn']['dataset']:
		# imagepath = basedata.CENTER_IMG_DIR + dataset + '_256x256/' + imgno + '.png'
		config = GMCNNOptions().parse(dataset=dataset, loadmodeldir=basedata.PRE_MODEL_DIR['gmcnn']['model'][dataset])
		config.dataset = dataset
		if dataset == 'places2':
			config.img_shapes = [int(512), int(680), int(3)]

		inputimgpath, outputimgpath = gmcnn_inpaint(imagepath, config)

		imagepath = imagepath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		outputimgpath = outputimgpath.split('static/')[-1]
		gmcnn_result = {
			'algrithm': 'gmcnn',
			'modelname': basedata.PRE_MODEL_DIR['gmcnn']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': outputimgpath,
			'truthimgpath': imagepath,
		}
		return gmcnn_result
	else:
		return False

def deepfill_center_inp(basedata,dataset,imagepath,inputimgpath,resultimgpath):
	if dataset in basedata.PRE_MODEL_DIR['deepfill']['dataset']:
		checkpointdir = basedata.PRE_MODEL_DIR['deepfill']['model'][dataset]

		maskpath = basedata.CENTER_MASK_256_DIR

		deepfill_inpaint(imagepath=imagepath, maskinfo=maskpath, status=1, checkpointdir=checkpointdir,
		                 inputimgpath=inputimgpath, outputpath=resultimgpath)

		imagepath = imagepath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		resultimgpath = resultimgpath.split('static/')[-1]
		deepfill_result = {
			'algrithm': 'deepfill',
			'modelname': basedata.PRE_MODEL_DIR['deepfill']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': resultimgpath,
			'truthimgpath': imagepath,
		}
		return deepfill_result
	else:
		return False

def conenc_center_inp(basedata,dataset,imagepath,inputimgpath,realimgpath,resultimgpath):
	if dataset in basedata.PRE_MODEL_DIR['conenc']['dataset']:

		loadmodeldir = basedata.PRE_MODEL_DIR['conenc']['model'][dataset]

		opt = ConEncOptions()
		opt.netG = loadmodeldir
		opt.test_image = imagepath
		opt.realimgpath = realimgpath
		opt.reconimgpath = resultimgpath
		opt.croppedimgpath = inputimgpath

		conenc_inp(opt)

		realimgpath = realimgpath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		resultimgpath = resultimgpath.split('static/')[-1]
		conenc_result = {
			'algrithm': 'conenc',
			'modelname': basedata.PRE_MODEL_DIR['conenc']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': resultimgpath,
			'truthimgpath': realimgpath,
		}
		return conenc_result


	else:
		return False

def edge_center_inp(basedata,dataset,imagepath,resultdir):

	if dataset in basedata.PRE_MODEL_DIR['edge']['dataset']:

		checkpointdir = basedata.PRE_MODEL_DIR['edge']['model'][dataset]
		maskimg = basedata.CENTER_MASK_256_DIR

		resultpaths = edgeinp(imagepath, maskimg, 1,checkpointdir, resultdir)
		if resultpaths:
			imagepath = imagepath.split('static/')[-1]
			inputimgpath = resultpaths[0].split('static/')[-1]
			resultimgpath = resultpaths[2].split('static/')[-1]
			conenc_result = {
				'algrithm': 'edge',
				'modelname': basedata.PRE_MODEL_DIR['edge']['modelname'][dataset],
				'inputimgpath': inputimgpath,
				'resultimgpath': resultimgpath,
				'truthimgpath': imagepath,
			}
			return conenc_result
		else:
			return False
	else:
		return False


def center_inp(basedata,dataset, imgno):
	context = {
		'resultstatus': 0,
		'dataset': dataset,
		'imgno': imgno,
		'inpaintresult': []
	}
	imagepath = os.path.join(basedata.CENTER_IMG_DIR, dataset + '_256x256/' + imgno + '.png')
	for algrithm in ['conenc','deepfill','edge','gmcnn']:
		res_dir = os.path.join(basedata.CENTER_RES_DIR, 'temp_' + dataset + '_'+algrithm+'')
		if os.path.exists(res_dir) is False:
			os.mkdir(res_dir)

		inputimgpath = os.path.join(res_dir, 'input_' + time.strftime('%Y%m%d%H%M%S') + imgno + '.png')
		realimgpath = os.path.join(res_dir, 'res_' + time.strftime('%Y%m%d%H%M%S') + imgno + '.png')
		resultimgpath = os.path.join(res_dir, 'real_' + time.strftime('%Y%m%d%H%M%S') + imgno + '.png')

		# run system use center mask with different algrithm model
		if algrithm=='gmcnn':
			gmcnn_result = gmcnn_center_inp(basedata,dataset, imagepath)
		elif algrithm == 'deepfill':
			deepfill_result = deepfill_center_inp(basedata,dataset,imagepath,inputimgpath,resultimgpath)
		elif algrithm == 'conenc':
			conenc_result = conenc_center_inp(basedata,dataset,imagepath,inputimgpath,realimgpath,resultimgpath)
		elif algrithm == 'edge':
			edge_result = edge_center_inp(basedata,dataset,imagepath,res_dir)

	if gmcnn_result:
		context['inpaintresult'].append(gmcnn_result)
	if deepfill_result:
		context['inpaintresult'].append(deepfill_result)
	if conenc_result:
		context['inpaintresult'].append(conenc_result)
	if edge_result:
		context['inpaintresult'].append(edge_result)

	if len(context['inpaintresult']) < 1:
		context['resultstatus'] = 0
	else:
		context['resultstatus'] = 1

	return context


def deepfill_freeform_inp(basedata,dataset, imagepath, masklocs):
	if dataset in basedata.PRE_MODEL_DIR['deepfill']['dataset']:
		checkpointdir = basedata.PRE_MODEL_DIR['deepfill']['model'][dataset]
		img_shapes = basedata.PRE_MODEL_DIR['gmcnn']['imageshape'][dataset]

		res_dir = os.path.join(basedata.CENTER_RES_DIR,'temp_' + dataset + '_deepfill')
		if os.path.exists(res_dir) is False:
			os.mkdir(res_dir)

		inputimgpath = os.path.join(res_dir,'input_' + time.strftime('%Y%m%d%H%M%S') + imagepath.split('/')[-1])
		resultimgpath = os.path.join(res_dir,'res_' + time.strftime('%Y%m%d%H%M%S') + imagepath.split('/')[-1])
		# print('thisisimagepath00000',basedata.UPLOAD_BASE_DIR,imagepath)
		imagepath = os.path.join(basedata.UPLOAD_BASE_DIR,(imagepath.split('upload/')[-1]))
		# print('thisisimagepathzzy',imagepath)
		deepfill_inpaint(img_shapes=img_shapes,imagepath=imagepath, maskinfo=masklocs, status=0, checkpointdir=checkpointdir,
		                 inputimgpath=inputimgpath, outputpath=resultimgpath)

		imagepath = imagepath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		resultimgpath = resultimgpath.split('static/')[-1]
		deepfill_result = {
			'algrithm': 'deepfill',
			'modelname': basedata.PRE_MODEL_DIR['deepfill']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': resultimgpath,
			'truthimgpath': imagepath,
		}
		return deepfill_result
	else:
		return False

def gmcnn_freeform_inp(basedata,dataset, imagepath, masklocs):
	if dataset in basedata.PRE_MODEL_DIR['gmcnn']['dataset']:
		imagepath = './App/static/' + imagepath
		config = GMCNNOptions().parse(dataset=dataset, loadmodeldir=basedata.PRE_MODEL_DIR['gmcnn']['model'][dataset])
		config.dataset = dataset
		config.img_shapes = basedata.PRE_MODEL_DIR['gmcnn']['imageshape'][dataset]

		inputimgpath, outputimgpath = gmcnn_inpaint(imagepath, config, masklocs)

		imagepath = imagepath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		outputimgpath = outputimgpath.split('static/')[-1]
		gmcnn_result = {
			'algrithm': 'gmcnn',
			'modelname': basedata.PRE_MODEL_DIR['gmcnn']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': outputimgpath,
			'truthimgpath': imagepath,
		}
		return gmcnn_result
	else:
		return False

def edge_freeform_inp(basedata,dataset,imagepath,masklocs):

	if dataset in basedata.PRE_MODEL_DIR['edge']['dataset']:

		res_dir = os.path.join(basedata.CENTER_RES_DIR, 'temp_' + dataset + '_deepfill')
		if os.path.exists(res_dir) is False:
			os.mkdir(res_dir)

		imagepath = os.path.join(basedata.UPLOAD_BASE_DIR, (imagepath.split('upload/')[-1]))
		checkpointdir = basedata.PRE_MODEL_DIR['edge']['model'][dataset]

		resultpaths = edgeinp(imagepath, masklocs,0, checkpointdir, res_dir)
		if resultpaths:
			imagepath = imagepath.split('static/')[-1]
			inputimgpath = resultpaths[0].split('static/')[-1]
			resultimgpath = resultpaths[2].split('static/')[-1]
			conenc_result = {
				'algrithm': 'edge',
				'modelname': basedata.PRE_MODEL_DIR['edge']['modelname'][dataset],
				'inputimgpath': inputimgpath,
				'resultimgpath': resultimgpath,
				'truthimgpath': imagepath,
			}
			return conenc_result
		else:
			return False
	else:
		return False

def exifill_freeform_inp(basedata,dataset, imagepath,prefillimgpath, masklocs):
	if (dataset in basedata.PRE_MODEL_DIR['exifill']['dataset']) and (prefillimgpath is not None):
		imagepath = './App/static/' + imagepath
		config = ExiFillOptions().parse(dataset=dataset, loadmodeldir=basedata.PRE_MODEL_DIR['exifill']['model'][dataset])
		# config.dataset = dataset
		# if dataset == 'places2':
		config.img_shapes = basedata.PRE_MODEL_DIR['exifill']['imageshape'][dataset]

		inputimgpath, outputimgpath = exifill_inpaint(basedata,imagepath, prefillimgpath,config, masklocs)

		imagepath = imagepath.split('static/')[-1]
		inputimgpath = inputimgpath.split('static/')[-1]
		outputimgpath = outputimgpath.split('static/')[-1]
		exifill_result = {
			'algrithm': 'exifill',
			'modelname': basedata.PRE_MODEL_DIR['exifill']['modelname'][dataset],
			'inputimgpath': inputimgpath,
			'resultimgpath': outputimgpath,
			'truthimgpath': imagepath,
		}
		return exifill_result
	else:
		return False


def freeform_inp(basedata,imagepath, dataset, rectmasks,algrithm=None,):
	rectmasks = [int(float(pix)) for pix in rectmasks.split(',')]
	masklocs = []
	for num in range(0, len(rectmasks) - 3, 4):
		onerect = []
		onerect.append(rectmasks[num])
		onerect.append(rectmasks[num + 1])
		onerect.append(rectmasks[num + 2])
		onerect.append(rectmasks[num + 3])
		masklocs.append(onerect)

	context = {
		'resultstatus': 0,
		'dataset': dataset,
		'inpaintresult': []
	}
	if algrithm is None or algrithm=='None':

		# run system use center mask with different algrithm model
		gmcnn_result = gmcnn_freeform_inp(basedata,dataset, imagepath, masklocs)
		deepfill_result = deepfill_freeform_inp(basedata,dataset, imagepath, masklocs)
		edge_result = edge_freeform_inp(basedata,dataset, imagepath, masklocs)

		if deepfill_result:
			prefillimgpath = './App/static/'+deepfill_result['resultimgpath']
		elif gmcnn_result:
			prefillimgpath = './App/static/'+gmcnn_result['resultimgpath']
		elif edge_result:
			prefillimgpath = './App/static/'+edge_result['resultimgpath']
		else:
			prefillimgpath = None
		exifill_result = exifill_freeform_inp(basedata,dataset,imagepath,prefillimgpath,masklocs)

		if gmcnn_result:
			context['inpaintresult'].append(gmcnn_result)
		if deepfill_result:
			context['inpaintresult'].append(deepfill_result)
		if edge_result:
			context['inpaintresult'].append(edge_result)
		if exifill_result:
			context['inpaintresult'].append(exifill_result)
	elif algrithm =='gmcnn':
		gmcnn_result = gmcnn_freeform_inp(basedata,dataset, imagepath, masklocs)
		if gmcnn_result:
			context['inpaintresult'].append(gmcnn_result)
	elif algrithm == 'deepfill':
		deepfill_result = deepfill_freeform_inp(basedata,dataset, imagepath, masklocs)
		if deepfill_result:
			context['inpaintresult'].append(deepfill_result)
	elif algrithm == 'edge':
		edge_result = edge_freeform_inp(basedata,dataset, imagepath, masklocs)
		if edge_result:
			context['inpaintresult'].append(edge_result)
	elif algrithm == 'exifil':
		deepfill_result = deepfill_freeform_inp(basedata, dataset, imagepath, masklocs)

		prefillimgpath = './App/static/' + deepfill_result['resultimgpath']
		exifill_result = exifill_freeform_inp(basedata, dataset, imagepath, prefillimgpath, masklocs)
		if exifill_result:
			context['inpaintresult'].append(exifill_result)

	if len(context['inpaintresult']) < 1:
		context['resultstatus'] = 0
	else:
		context['resultstatus'] = 1

	return context
