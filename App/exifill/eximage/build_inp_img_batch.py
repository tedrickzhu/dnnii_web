# -*- coding: utf-8 -*-
# @Time    : 20-3-3 下午12:39
# @Author  : zhuzhengyi

from deepfill.deepfill_batch import deepfillbatch

if __name__ == '__main__':
	# parser = argparse.ArgumentParser()
	#step : 0,recutimages;1,create train txt file;2,create inpainted image
	# parser.add_argument('--step', default=-1, type=int,help='choosed step of dataprepare.')
	# parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	# parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	# parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	# parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	# parser.add_argument('--recutdir', default='', type=str,help='The saved dir of recutted images.')
	# parser.add_argument('--neednums', default=10, type=int,help='The total numbers of recutted images.')
	# args = parser.parse_args()

	# recutimage(args.dataset,args.recutdir,args.neednums)
	# print('======recut images finished======')

	# baserecutpath = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680/'
	# maskpath = "/home/zzy/work/experiment_image_inpaint/searcher/examples/center_mask_512x680_200.png"
	# maskpath = "/home/zzy/work/experiment_image_inpaint/searcher/examples/center_mask_512x680_128.png"
	# img_mask_txt_path = './deepfill/img_mask_path128.txt'
	img_mask_txt_path = './deepfill/img_mask_path200.txt'
	# if not os.path.isfile(img_mask_txt_path):
	# create_datafile(baserecutpath,maskpath,img_mask_txt_path)
	#
	checkpointdir = "/home/zzy/work/experiment_image_inpaint/searcher/deepfill/checkpoints/places2_512x680"
	outputdir = '/home/zzy/work/experiment_image_inpaint/searcher/examples/output'
	deepfillbatch(512, 680, checkpointdir, img_mask_txt_path, outputdir)
	#
	# print('finished')

