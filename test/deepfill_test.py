# -*- coding: utf-8 -*-
# @Time    : 20-2-23 上午12:15
# @Author  : zhuzhengyi
from App.conenc.conenc_options import ConEncOptions
from App.conenc.mytest import conenc_inp
from App.deepfill.mytest import deepfill_inpaint
import glob,os

def deepfillceshi():
    pretrained_models={
        "celebahq":"../App/checkpoints/deepfill/celebahq_256",
        "places2":"../App/checkpoints/deepfill/places2_256"
    }
    basedir = '/home/zzy/TrainData/MITPlace2Dataset/badsource/'
    filelist = glob.glob(basedir+'*.png')
    # print(filelist)
    checkpointdir=pretrained_models["places2"]

    # imagepath = "../App/static/images/celeba_256x256/001.png"
    maskpath="./center_mask_256_100.png"
    for imagepath in filelist:
        inputimgpath = './tmp_res/deepfill_input/'+os.path.basename(imagepath)
        outputpath = './tmp_res/deepfill_res/'+os.path.basename(imagepath)
        deepfill_inpaint((256,256),imagepath, maskpath,1, checkpointdir, inputimgpath,outputpath)

    print("this is done")

def contextencodertest():
    checkpointdir = "/home/zzy/work/dnnii_web/dnnii_web/App/checkpoints/conenc/netG_streetview.pth"
    basedir = '/home/zzy/TrainData/MITPlace2Dataset/badsource/'
    filelist = glob.glob(basedir + '*.png')
    maskpath = "./center_mask_256_100.png"

    opt = ConEncOptions()
    opt.netG = checkpointdir

    for imagepath in filelist:
        realimgpath = './tmp_res/context_truth/' + os.path.basename(imagepath)
        inputimgpath = './tmp_res/context_input/' + os.path.basename(imagepath)
        outputpath = './tmp_res/context_res/' + os.path.basename(imagepath)
        opt.test_image = imagepath
        opt.realimgpath = realimgpath
        opt.reconimgpath = outputpath
        opt.croppedimgpath = inputimgpath
        conenc_inp(opt)
        # deepfill_inpaint((256, 256), imagepath, maskpath, 1, checkpointdir, inputimgpath, outputpath)

    print("this is done")

if __name__ == '__main__':
    # contextencodertest()
    deepfillceshi()