import tensorflow as tf

class DataLoader:
    def __init__(self, imgfilepath,eximgfilepath, im_size, batch_size):
        self.imgfilelist = open(imgfilepath, 'rt').read().splitlines()
        self.eximgfilelist = open(eximgfilepath, 'rt').read().splitlines()

        if not self.imgfilelist:
            exit('\nError: file list is empty\n')
        
        self.im_size = im_size
        self.batch_size = batch_size
        self.dataimg_queue = None
        self.dataeximg_queue = None
        self.data_queue = None

    def next(self):
        with tf.variable_scope('feed'):
            imgfilelist_tensor = tf.convert_to_tensor(self.imgfilelist, dtype=tf.string)
            eximgfilelist_tensor = tf.convert_to_tensor(self.eximgfilelist, dtype=tf.string)
            # self.data_queue = tf.train.slice_input_producer([imgfilelist_tensor])
            self.dataimg_queue, self.dataeximg_queue = tf.train.slice_input_producer([imgfilelist_tensor,eximgfilelist_tensor])

            # ttim_gt = tf.image.decode_image(tf.read_file(self.data_queue[0]), channels=3)
            im_gt = tf.image.decode_image(tf.read_file(self.dataimg_queue), channels=3)
            exim_gt = tf.image.decode_image(tf.read_file(self.dataeximg_queue), channels=3)

            # im_gt = tf.cast(im_gt, tf.float32) / 127.5 - 1
            im_gt = tf.cast(im_gt, tf.float32)
            exim_gt = tf.cast(exim_gt, tf.float32)

            im_gt = tf.image.resize_image_with_crop_or_pad(im_gt, self.im_size[0], self.im_size[1])
            exim_gt = tf.image.resize_image_with_crop_or_pad(exim_gt, self.im_size[0], self.im_size[1])

            im_gt.set_shape([self.im_size[0], self.im_size[1], 3])
            exim_gt.set_shape([self.im_size[0], self.im_size[1], 3])

            ttbatch_gt = tf.train.batch([im_gt], batch_size=self.batch_size, num_threads=4)

            batch_gt = tf.train.batch([im_gt,exim_gt], batch_size=self.batch_size, num_threads=4)
        return batch_gt
