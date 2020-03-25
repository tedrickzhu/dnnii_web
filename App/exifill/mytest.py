import time

import numpy as np
import cv2
import os
import glob
import tensorflow as tf
from App.exifill.exigmcnn.util import generate_mask_rect, generate_mask_stroke
from App.exifill.exigmcnn.network import GMCNNModel

from App.exifill.eximage.util.build_img_pairs import get_img_pair

# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmax([int(x.split()[2]) for x in subprocess.Popen(
#         "nvidia-smi -q -d Memory | grep -A4 GPU | grep Free", shell=True, stdout=subprocess.PIPE).stdout.readlines()]
#         ))
from App.util.util import resize_img_eximg


def exifill_inpaint(basedata,imagepath,prefillimgpath, config, masklocs):
    #get similirity image according to prefillimg
    parameters = [prefillimgpath,basedata.COLORFEATURESCSV,basedata.STRUCTUREFEATURESCSV]
    repairlist = get_img_pair(parameters)

    if repairlist[1]==os.path.basename(imagepath):
        eximgpath = imagepath
    else:
        print(type(basedata.BASEDATASET),basedata.BASEDATASET)
        eximgpath = basedata.BASEDATASET+repairlist[1]+'.png'
    # eximgpath = imagepath
    # print('this is imagepath',imagepath)
    model = GMCNNModel()
    print('this is after model create')
    reuse = False
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = False
    with tf.Session(config=sess_config) as sess:
        input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
        input_eximage_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
        input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

        output = model.evaluate(input_image_tf, input_eximage_tf,input_mask_tf, config=config, reuse=tf.AUTO_REUSE)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
        output = tf.cast(output, tf.uint8)

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                              vars_list))
        sess.run(assign_ops)
        print('Model loaded.')
        # total_time = 0
        image = cv2.imread(imagepath)
        eximage = cv2.imread(eximgpath)

        if config.random_mask:
            np.random.seed(config.seed)

        if masklocs is not None:
            mask = np.zeros((config.img_shapes[0], config.img_shapes[1], 1)).astype(np.uint8)
            for rect in masklocs:
                mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] = 1
        else:
            # mask = generate_mask_rect(image.shape, config.mask_shapes, config.random_mask)
            mask = generate_mask_rect(config.img_shapes, config.mask_shapes, config.random_mask)

        image = resize_img_eximg(image,config.img_shapes)
        eximage = resize_img_eximg(eximage,config.img_shapes)

        # cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image.astype(np.uint8))
        image = image * (1-mask) + 255 * mask
        inputimgpath = os.path.join(config.saving_path,'input' + time.strftime('%Y%m%d%H%M%S') + '_' + imagepath.split("/")[-1])
        cv2.imwrite(inputimgpath, image.astype(np.uint8))

        assert image.shape[:2] == mask.shape[:2]

        h, w = image.shape[:2]
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        eximage = eximage[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        eximage = np.expand_dims(eximage, 0)
        mask = np.expand_dims(mask, 0)
        # 将图片和蒙版图同时输入网络中，得到结果============================================
        result = sess.run(output, feed_dict={input_image_tf: image, input_eximage_tf:eximage, input_mask_tf: mask})

        outputimgpath = os.path.join(config.saving_path,
                                     "res_" + time.strftime('%Y%m%d%H%M%S') + '_' + imagepath.split("/")[-1])
        cv2.imwrite(outputimgpath, result[0][:, :, ::-1])

        sess.close()

        return inputimgpath, outputimgpath
