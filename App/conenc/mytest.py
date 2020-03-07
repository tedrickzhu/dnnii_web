from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable

from App.conenc.model import _netG
from App.conenc.conenc_options import ConEncOptions
from App.conenc import utils

'''
context_encoder 模型使用说明
1，目前仅有paris_streetview这一数据集训练的模型
2，调用此函数前需变更以下参数为自己的路径
    opt.test_image,//原始图片位置
    opt.realimgpath,//变更成模型尺寸后的真实图片，需要保存的位置
    opt.croppedimgpath,//加入中间的白框的输入图片的保存位置
    opt.reconimgpath//修复结果的保存位置

'''

def conenc_inp(opt):

    netG = _netG(opt)
    # netG = TransformerNet()
    netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    # netG.requires_grad = False
    netG.eval()

    transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = utils.load_image(opt.test_image, opt.imageSize)
    image = transform(image)
    image = image.repeat(1, 1, 1, 1)

    input_real = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
    input_cropped = torch.FloatTensor(1, 3, opt.imageSize, opt.imageSize)
    real_center = torch.FloatTensor(1, 3, int(opt.imageSize/2), int(opt.imageSize/2))

    criterionMSE = nn.MSELoss()

    # if opt.cuda:
    #     netG.cuda()
    #     input_real, input_cropped = input_real.cuda(),input_cropped.cuda()
    #     criterionMSE.cuda()
    #     real_center = real_center.cuda()

    input_real = Variable(input_real)
    input_cropped = Variable(input_cropped)
    real_center = Variable(real_center)


    input_real.data.resize_(image.size()).copy_(image)
    input_cropped.data.resize_(image.size()).copy_(image)
    real_center_cpu = image[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)]
    real_center.data.resize_(real_center_cpu.size()).copy_(real_center_cpu)

    input_cropped.data[:,0,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*117.0/255.0 - 1.0
    input_cropped.data[:,1,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*104.0/255.0 - 1.0
    input_cropped.data[:,2,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred,int(opt.imageSize/4)+opt.overlapPred:int(opt.imageSize/4)+int(opt.imageSize/2)-opt.overlapPred] = 2*123.0/255.0 - 1.0

    print(type(input_cropped),input_cropped.shape)
    fake = netG(input_cropped)
    print(type(fake),type(real_center),fake.shape,real_center.shape)
    errG = criterionMSE(fake,real_center)

    recon_image = input_cropped.clone()
    recon_image.data[:,:,int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2),int(opt.imageSize/4):int(opt.imageSize/4)+int(opt.imageSize/2)] = fake.data

    utils.save_image(opt.realimgpath,image[0])
    utils.save_image(opt.croppedimgpath,input_cropped.data[0])
    utils.save_image(opt.reconimgpath,recon_image.data[0])

    # print('%.4f' % errG.item())

if __name__ == '__main__':
    opt= ConEncOptions()
    conenc_inp(opt)