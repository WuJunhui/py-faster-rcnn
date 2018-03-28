
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detectcls
#from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
import numpy as np
#import scipy.io as sio
import caffe, os, sys, cv2
import argparse

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel')}

def vis_detections(im ):
    """Draw detected bounding boxes."""
    imshow=im.copy()
    cv2.imshow('imshow',imshow)


def demo(net, image_name, vis=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    #print 'test im.shape'
    #print im.shape

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    cls_prob = im_detectcls(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, cls_prob.shape[0])
    if vis:
        vis_detections(im )
    return cls_prob

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    cfg.TEST.SCALE_MULTIPLE_OF = 32
    cfg.TEST.SCALES=(352,)
    args = parse_args()

    rootdir ='/export/home/wjh/py-faster-rcnn/'
    prototxt = rootdir + 'workdir_merge/testmerge.prototxt'
    caffemodel = rootdir + 'workdir_merge/output/voccls_2018_train/zf_faster_rcnn_iter_670000.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(1):
        _= im_detectcls(net, im)
    imdir='/export/home/wjh/project_npx/data/'
    annofile=imdir + 'val.txt'

    anno=np.loadtxt(annofile,dtype='str')
    vis=False
    outf=open('outclsmerge.txt','w')
    for i in range(anno.shape[0]):
        im_name=anno[i,0]
        gt_label=anno[i,1].astype(int)
        im_path=imdir+im_name
        print 'Demo for {} {}'.format(im_name,gt_label)
        prob=demo(net, im_path, vis)
        outline=' '.join([im_name, str(gt_label), str(prob[0][0]), str(prob[0][1])])
        print outline
        outf.write(outline+'\n')
        #cv2.waitKey(0)
