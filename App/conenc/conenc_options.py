# -*- coding: utf-8 -*-
# @Time    : 20-2-14 下午9:16
# @Author  : zhuzhengyi

'''
parser.add_argument('--dataset',  default='streetview', help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--test_image', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

parser.add_argument('--nBottleneck', type=int,default=4000,help='of dim for bottleneck of encoder')
parser.add_argument('--overlapPred',type=int,default=4,help='overlapping edges')
parser.add_argument('--nef',type=int,default=64,help='of encoder filters in first conv layer')
parser.add_argument('--wtl2',type=float,default=0.999,help='0 means do not use else use with this weight')
'''

class ConEncOptions:
    def __init__(self):
        self.dataset='paris'
        #test image path
        self.test_image='072_im.png'
        self.workers=4
        self.batchsize=64
        self.imageSize=128
        self.nz=100
        self.ngf=64
        self.ndf=64
        self.nc=3
        self.niter=25
        self.lr=0.0002
        self.beta1=0.5
        self.cuda=None
        self.ngpu=1
        self.netG="checkpoints/netG_streetview.pth"
        self.netD=""
        self.outf="./"
        self.manualSeed=None
        self.nBottleneck=4000
        self.overlapPred=4
        self.nef=64
        self.wtl2=0.999
        self.realimgpath = 'val_real_samples.png'
        self.croppedimgpath = 'val_cropped_samples.png'
        self.reconimgpath = 'val_recon_samples.png'
