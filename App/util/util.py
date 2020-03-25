# -*- coding: utf-8 -*-
# @Time    : 20-3-8 上午12:54
# @Author  : zhuzhengyi

import cv2
import numpy as np

def resize_img_eximg(image,img_shapes):
    h,w,c= image.shape[:]
    if h >= img_shapes[0] and w >= img_shapes[1]:
        h_start = (h - img_shapes[0]) // 2
        w_start = (w - img_shapes[1]) // 2
        image = image[h_start: h_start + img_shapes[0], w_start: w_start + img_shapes[1], :]
    else:
        t = min(h, w)
        image = image[(h - t) // 2:(h - t) // 2 + t, (w - t) // 2:(w - t) // 2 + t, :]
        # print(image.shape)
        image = cv2.resize(image, (img_shapes[1], img_shapes[0]))
        # print(image.shape)
        if len(image.shape)<3:
            image = np.expand_dims(image, axis=2)
    return image
