#encoding=utf-8
import numpy as np
import cv2
import os
import time
import tensorflow as tf
from App.gmcnn.util.util import generate_mask_rect
from App.gmcnn.net.network import GMCNNModel
from App.util.util import resize_img_eximg

'''
GMCNN 输入图片和mask说明
1，输入的图片为truth图片
2，输入的mask图片中含有待修复区域的信息，choosed area 为1，其他像素值为0
3，gmcnn 可处理的mask模式（mask 的区域不应超过整个图片面积的20%~30%）：
    1)一个center rectangle mask
    2)多个rectangle mask
    3)random choosed area mask(任意区域，任意形状) 
'''

def gmcnn_inpaint(imagepath,config,masklocs=None):
    model = GMCNNModel()

    # reuse = False
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = False
    gmcnn_graph = tf.Graph()
    with tf.Session(config=sess_config,graph=gmcnn_graph) as gmcnn_sess:
        input_image_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 3])
        input_mask_tf = tf.placeholder(dtype=tf.float32, shape=[1, config.img_shapes[0], config.img_shapes[1], 1])

        output = model.evaluate(input_image_tf, input_mask_tf, config=config, reuse=tf.AUTO_REUSE)
        output = (output + 1) * 127.5
        output = tf.minimum(tf.maximum(output[:, :, :, ::-1], 0), 255)
        output = tf.cast(output, tf.uint8)

        # load pretrained model
        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = list(map(lambda x: tf.assign(x, tf.contrib.framework.load_variable(config.load_model_dir, x.name)),
                              vars_list))
        gmcnn_sess.run(assign_ops)
        print('gmcnn Model loaded.')
        total_time = 0

        if config.random_mask:
            np.random.seed(config.seed)

        # 放缩调整输入图片的尺寸，和输入的蒙版图的尺寸==========================================================
        image = cv2.imread(imagepath)
        h, w = image.shape[:2]

        if masklocs is not None:
            mask = np.zeros((h, w, 3)).astype(np.uint8)
            for rect in masklocs:
                mask[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2], :] = 1
        else:
            mask = generate_mask_rect(config.img_shapes, config.mask_shapes, config.random_mask)

        image = resize_img_eximg(image,config.img_shapes)
        mask = resize_img_eximg(mask,config.img_shapes)
        mask = mask[:,:,0:1]

        # cv2.imwrite(os.path.join(config.saving_path, 'gt_{:03d}.png'.format(i)), image.astype(np.uint8))
        image = image * (1-mask) + 255 * mask
        inputimgpath = os.path.join(config.saving_path, 'input'+time.strftime('%Y%m%d%H%M%S')+'_' + imagepath.split("/")[-1])
        cv2.imwrite(inputimgpath, image.astype(np.uint8))

        assert image.shape[:2] == mask.shape[:2]

        h, w = image.shape[:2]
        grid = 4
        image = image[:h // grid * grid, :w // grid * grid, :]
        mask = mask[:h // grid * grid, :w // grid * grid, :]

        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)

        #将图片和蒙版图同时输入网络中，得到结果============================================
        result = gmcnn_sess.run(output, feed_dict={input_image_tf: image, input_mask_tf: mask})

        outputimgpath = os.path.join(config.saving_path, "res_"+time.strftime('%Y%m%d%H%M%S')+'_' + imagepath.split("/")[-1])
        cv2.imwrite(outputimgpath, result[0][:, :, ::-1])

        gmcnn_sess.close()

        return inputimgpath, outputimgpath
