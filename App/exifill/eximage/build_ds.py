# -*- coding: utf-8 -*-
# @Time    : 20-3-3 下午12:39
# @Author  : zhuzhengyi

import argparse
from util.build_dataset import recutimage

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	parser.add_argument('--recutdir', default='', type=str,help='The saved dir of recutted images.')
	parser.add_argument('--readnums', default=0, type=int,help='The total numbers of recutted images.')
	args = parser.parse_args()
	 
	recutimage(args.dataset,args.recutdir,args.readnums)
	print('======recut images finished======')


