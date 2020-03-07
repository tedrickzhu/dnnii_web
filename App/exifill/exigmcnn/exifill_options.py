import argparse
import os
import time

class ExiFillOptions:
    def __init__(self):
        self.parser = TestArgs()


    def parse(self,dataset="places2",loadmodeldir='./gmcnn/checkpoints/paris-streetview_256x256_rect'):

        self.opt = self.parser

        if os.path.exists(self.opt.test_dir) is False:
            os.mkdir(self.opt.test_dir)

        assert self.opt.random_mask in [0, 1]
        self.opt.random_mask = True if self.opt.random_mask == 1 else False

        assert self.opt.mask_type in ['rect', 'stroke']

        str_img_shapes = self.opt.img_shapes.split(',')
        self.opt.img_shapes = [int(x) for x in str_img_shapes]

        str_mask_shapes = self.opt.mask_shapes.split(',')
        self.opt.mask_shapes = [int(x) for x in str_mask_shapes]

        self.opt.dataset = dataset
        self.opt.load_model_dir = loadmodeldir

        # saving dir name form = model name and date
        self.opt.model_folder = 'temp_' + self.opt.dataset + '_' + self.opt.model
        # self.opt.model_folder += '_s' + str(self.opt.img_shapes[0]) + 'x' + str(self.opt.img_shapes[1])
        # self.opt.model_folder += '_gc' + str(self.opt.g_cnum)
        # self.opt.model_folder += '_randmask-' + self.opt.mask_type if self.opt.random_mask else ''
        if self.opt.random_mask:
            self.opt.model_folder += '_seed-' + str(self.opt.seed)
        self.opt.saving_path = os.path.join(self.opt.test_dir, self.opt.model_folder)

        if os.path.exists(self.opt.saving_path) is False and self.opt.mode == 'save':
            os.mkdir(self.opt.saving_path)

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt

class TestArgs:
    def __init__(self):
        self.dataset='paris'
        self.data_file='./App/static/images/pstreetview_256x256'
        self.test_dir='./App/static/center_results'
        self.load_model_dir='./App/gmcnn/checkpoints/paris-streetview_256x256_rect'
        self.model_prefix='snap'
        self.seed=1
        self.model='gmcnn'
        self.img_shapes='256,256,3'

        self.mask_shapes='128,128'
        self.mask_type='rect'
        self.random_mask=0

        self.test_num=-1
        self.mode='save'
        # for generator
        self.g_cnum=32
        self.d_cnum=64