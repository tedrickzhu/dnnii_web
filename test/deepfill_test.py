# -*- coding: utf-8 -*-
# @Time    : 20-2-23 上午12:15
# @Author  : zhuzhengyi
from App.deepfill.mytest import deepfill_inpaint


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
