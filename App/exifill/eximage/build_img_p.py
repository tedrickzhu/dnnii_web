# -*- coding: utf-8 -*-
# @Time    : 20-3-3 下午12:39
# @Author  : zhuzhengyi

import argparse
from util.build_img_pairs import buildimgpairs

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', default='', type=str,help='The dir of images to be recutted.')
	parser.add_argument('--colorf', default='', type=str,help='the dir saved color features of dataset.')
	parser.add_argument('--structf', default='', type=str,help='the dir saved struct features of dataset.')
	parser.add_argument('--imgfile', default='', type=str,help='the filepath of images to save')
	parser.add_argument('--eximgfile', default='', type=str,help='the filepath of ex images to save')
	parser.add_argument('--fdbdir', default=None, type=str,help='the filepath of dataset which build features db based on.')
	args = parser.parse_args()
	if args.fdbdir is None:
		buildimgpairs(args.dataset, args.colorf, args.structf, args.imgfile, args.eximgfile)
	else:
		buildimgpairs(args.dataset, args.colorf, args.structf, args.imgfile, args.eximgfile,args.fdbdir)
	print('======buildimgpairs finished======')

