# -*- coding: utf-8 -*-
# @Time    : 20-3-4 上午2:01
# @Author  : zhuzhengyi
import argparse
import glob

def create_datafile(datasetdir,outputpath):
    with open(outputpath,'a+') as txtfile:
        # for index in range(1,imagesnums):
        for imagepath in glob.glob(datasetdir + "*.png"):
            # imagename = 'Places365_' + str(index).zfill(8) + '.png'
            # imagepath = os.path.join(baserecutpath, imagename)
            line = str(imagepath)
            txtfile.write(line)
            txtfile.write('\r\n')
    txtfile.close()
    print('finished')

if __name__ == '__main__':
    # datasetdir = '/home/zzy/TrainData/MITPlace2Dataset/val_recut_512x680/'
    # outputpath = '/home/zzy/work/exi_inpaint/places2_512x680_file.txt'
    # create_datafile(datasetdir,outputpath)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='', type=str, help='The dir of images to be recutted.')
    parser.add_argument('--output', default='', type=str, help='the dir saved color features of dataset.')
    args = parser.parse_args()
    create_datafile(args.dataset, args.output)
