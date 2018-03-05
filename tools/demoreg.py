
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
from fast_rcnn.test import im_detectreg
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

def vis_detections(im, boxes,gt_bbox):
    """Draw detected bounding boxes."""

    imshow=im.copy()
    bbox = boxes[0]
    cv2.rectangle(imshow,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),(0,0,255),2)
    cv2.rectangle(imshow,(int(gt_bbox[0]),int(gt_bbox[1])),(int(gt_bbox[2]),int(gt_bbox[3])),(255,255,255),2)
    cv2.imshow('imshow',imshow)

def bb_iou(boxA,boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    # bboxA[xmin,ymin,xmax,ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0.0,(xB - xA + 1)) * max(0.0,(yB - yA + 1))
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def demo(net, image_name, gt_bbox, vis=False):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = image_name
    im = cv2.imread(im_file)
    print 'test im.shape'
    print im.shape

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    boxes = im_detectreg(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    if vis:
        vis_detections(im, boxes, gt_bbox)
    iou=bb_iou(boxes[0],gt_bbox)
    return boxes,gt_bbox,iou

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
    prototxt = rootdir + 'workdir_fg/testreg.prototxt'
    caffemodel = rootdir + 'workdir_fg/output/zf_faster_rcnn_iter_200000.caffemodel'
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
        _= im_detectreg(net, im)
    imdir='/export/home/wjh/multi-task/fgtest/JPEGImages/'
    annofile='/export/home/wjh/multi-task/fgtest/test.bbox'

    anno=np.loadtxt(annofile,dtype='str')
    vis=False
    ious=[]
    for i in range(anno.shape[0]):
    #for i in range(100):
        im_name=anno[i,0]
        gt_bbox=anno[i,1:].astype(float)

        im_path=imdir+im_name
        print 'Demo for {}'.format(im_name)
        bboxs,gt_bbox,iou=demo(net, im_path, gt_bbox, vis)
        print bboxs, gt_bbox,iou
        ious.append(iou)
        #cv2.waitKey(0)

    print np.mean(ious)

