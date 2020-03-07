# -*- coding: utf-8 -*-
# @Time    : 20-2-26 下午10:53
# @Author  : zhuzhengyi

class BaseData:
	def __init__(self):
		self.CONENC_BASE_DIR = './App/conenc/'
		self.DEEPFILL_BASE_DIR = './App/deepfill/'
		self.EDGE_BASE_DIR = './App/edge/'
		self.GMCNN_BASE_DIR = './App/gmcnn/'

		self.CHECKPOINTS_BASE_DIR = './App/checkpoints/'

		self.CENTER_MASK_256_DIR = './App/static/images/maskimg/center_mask_256.png'
		self.CENTER_IMG_DIR = './App/static/images/'
		self.CENTER_RES_DIR = './App/static/center_results/'
		self.UPLOAD_BASE_DIR = './App/static/upload/'

		self.COLORFEATURESCSV = '/home/zzy/work/dnnii_web/dnnii_web/App/exifill/eximage/files/b500colorfeatures.csv'
		self.STRUCTUREFEATURESCSV = '/home/zzy/work/dnnii_web/dnnii_web/App/exifill/eximage/files/b500strucfeatures.csv'
		self.BASEDATASET = '/home/zzy/TrainData/MITPlace2Dataset/base500recut/'

		self.ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'jpeg', 'JPEG'])

		self.PRE_MODEL_DIR = {
			'gmcnn': {
				'dataset': ['pstreetview', 'celebahq', 'places2'],
				'model': {'pstreetview': './App/checkpoints/gmcnn/paris-streetview_256x256_rect',
				          'celebahq': './App/checkpoints/gmcnn/celebahq_256x256_rect',
				          'places2': './App/checkpoints/gmcnn/places2_512x680_freeform'
				          },
				'modelname': {'pstreetview': 'paris-streetview_256_rect',
				              'celebahq': 'celebahq_256_rect',
				              'places2': 'places2_512_freeform'
				              },
				'imageshape': {'pstreetview': [256,256,3],
				              'celebahq': [256,256,3],
				              'places2': [512,680,3]
				              }
			},
			'deepfill': {
				'dataset': ['places2', 'celebahq'],
				'model': {'places2': './App/checkpoints/deepfill/places2_256',
				          'celebahq': './App/checkpoints/deepfill/celebahq_256'
				          },
				'modelname': {'places2': 'places2_256_freeform',
				              'celebahq': 'celebahq_256_freeform'
				              },
				'imageshape': {'celebahq': [256,256,3],
				              'places2': [256,256,3]
				              }
			},
			'conenc': {
				'dataset': ['pstreetview'],
				'model': {'pstreetview': './App/checkpoints/conenc/netG_streetview.pth'},
				'modelname': {'pstreetview': 'pstreetview-center-128'},
				'imageshape': {'pstreetview': [256,256,3]}
			},
			'edge': {
				'dataset': ['pstreetview', 'celeba','celebahq', 'places2'],
				'model': {'pstreetview': './App/checkpoints/edge/psv',
				          'celebahq': './App/checkpoints/edge/celeba',
				          'places2': './App/checkpoints/edge/places2'
				          },
				'modelname': {'pstreetview': 'paris-streetview',
				              'celebahq': 'celeba',
				              'places2': 'places2'
				              },
				'imageshape': {'pstreetview': [256,256,3],
				              'celebahq': [256,256,3],
				              'places2': [256,256,3]
				              }
			},
			'exifill': {
				'dataset': ['places2'],
				'model': {'pstreetview': './App/checkpoints/edge/psv',
				          'celebahq': './App/checkpoints/edge/celeba',
				          'places2': './App/checkpoints/exifill/places2_512'
				          },
				'modelname': {'pstreetview': 'paris-streetview',
				              'celebahq': 'celeba',
				              'places2': 'places2'
				              },
				'imageshape': {'pstreetview': [256,256,3],
				              'celebahq': [256,256,3],
				              'places2': [512,680,3]
				              }
			}
		}