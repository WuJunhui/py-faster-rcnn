# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import caffe
#import yaml
import numpy as np
#import numpy.random as npr
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform
#from utils.cython_bbox import bbox_overlaps

DEBUG = False

class OneProposalTargetLayer(caffe.Layer):
    """
    Assign object detection proposals to ground-truth targets. Produces proposal
    classification labels and bounding-box regression targets.
    """

    def setup(self, bottom, top):
        #layer_params = yaml.load(self.param_str_)
        #self._num_classes = layer_params['num_classes']
        self._num_classes = 1

        # sampled rois (0, x1, y1, x2, y2)
        top[0].reshape(1, 5)
        # labels
        #top[1].reshape(1, 1)
        # bbox_targets
        top[1].reshape(1, self._num_classes * 4)
        # bbox_inside_weights
        top[2].reshape(1, self._num_classes * 4)
        # bbox_outside_weights
        top[3].reshape(1, self._num_classes * 4)

    def forward(self, bottom, top):
        # Proposal ROIs (0, x1, y1, x2, y2) coming from RPN
        # (i.e., rpn.proposal_layer.ProposalLayer), or any other source
        all_rois = bottom[0].data
        # GT boxes (x1, y1, x2, y2, label)
        # TODO(rbg): it's annoying that sometimes I have extra info before
        # and other times after box coordinates -- normalize to one format
        gt_boxes = bottom[1].data

        # Include ground-truth boxes in the set of candidate rois
        #zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        #all_rois = np.vstack(
        #    (all_rois, np.hstack((zeros, gt_boxes[:, :-1])))
        #)

        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), \
                'Only single item batches are supported'

        #num_images = 1
        #rois_per_image = cfg.TRAIN.BATCH_SIZE / num_images
        #fg_rois_per_image = np.round(cfg.TRAIN.FG_FRACTION * rois_per_image)

        # Sample rois with classification labels and bounding box regression
        # targets
        rois, bbox_targets, bbox_inside_weights = _sample_rois(
            all_rois, gt_boxes)
            #all_rois, gt_boxes, fg_rois_per_image,
            #rois_per_image, self._num_classes)


        # sampled rois
        top[0].reshape(*rois.shape)
        top[0].data[...] = rois


        # classification labels
        #top[1].reshape(*labels.shape)
        #top[1].data[...] = labels

        # bbox_targets
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets

        # bbox_inside_weights
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights


        # bbox_outside_weights
        top[3].reshape(*bbox_inside_weights.shape)
        top[3].data[...] = np.array(bbox_inside_weights > 0).astype(np.float32)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass


def _get_bbox_regression_labels(bbox_target_data):
    """Bounding-box regression targets (bbox_target_data) are stored in a
    compact form N x (class, tx, ty, tw, th)

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets).

    Returns:
        bbox_target (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    bbox_targets = bbox_target_data[:,1:]
    bbox_inside_weights = np.ones_like(bbox_targets)
    return bbox_targets, bbox_inside_weights

def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 4

    targets = bbox_transform(ex_rois, gt_rois)



    if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
        # Optionally normalize targets by a precomputed mean and stdev
        targets = ((targets - np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS))
                / np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS))
    labels=np.ones((targets.shape[0]))
    return np.hstack(
            (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)

def _sample_rois(all_rois, gt_boxes):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois = all_rois.copy()
    #===wjh in case gt_boxes num is more than 1
    bbox_target_data = _compute_targets(
            rois[:, 1:5], gt_boxes[0:1, :4])

    bbox_targets, bbox_inside_weights = \
        _get_bbox_regression_labels(bbox_target_data)
    return rois, bbox_targets, bbox_inside_weights
