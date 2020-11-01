#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np



def display_with_bbox(img, probs, bboxes, nmax=10, vmin=0, vmax=250, show_text = True, color = 'r'):
    '''
    Display the img with bounding boxes
    
    @params
    @img: 2-dimensional np.array, the image to be displayed
    @probs: list of probabilities of each bound box
    @bboxes: list of length-4 lists, each lists contains [x,y,width,height]
    @nmax: maximum number of boxes to be displayed
    @vmin: vmin of image display
    @vmax: vmax of image display
    @show_text: if True then shows the probability on the bbox
    @color: the color of the bounding boxes
    '''
    if probs is None:
        probs = np.ones([len(bboxes)], np.float32)
    
    plt.imshow(img, 'gray', vmin=vmin, vmax=vmax)
    if len(probs) > nmax:
        probs = probs[:nmax]
        bboxes = bboxes[:nmax]
    
    for prob, bbox in zip(probs, bboxes):
        rect = patches.Rectangle([bbox[0],bbox[1]], bbox[2], bbox[3], linewidth=1, edgecolor=color, facecolor='none', clip_on = False)
        plt.gca().add_patch(rect)
        if show_text:
            plt.text(bbox[0]+6,bbox[1]-6, '%.3f'%prob, color='w', bbox=dict(facecolor=color, alpha=0.5))


def IoU(target, bboxes):
    '''
    Calculate the intersection over union for bboxes over target
    
    @params:
    @target: a list/tuple/array, a bounding box of [x,y,width,height]
    @bboxes: a list of length N, bounding boxes each with [x,y,width,height]
    
    @returns:
    @ious: array of length N, the ious of the target with each bbox
    '''
    
    bboxes = np.array(bboxes)
    
    # calculate intersection
    start_x = np.maximum(bboxes[...,0], target[0])
    end_x = np.minimum(bboxes[...,0] + bboxes[...,2], target[0] + target[2])
    start_y = np.maximum(bboxes[...,1], target[1])
    end_y = np.minimum(bboxes[...,1] + bboxes[...,3], target[1] + target[3])
    
    ind_x = np.where(end_x < start_x)
    end_x[ind_x] = start_x[ind_x]
    
    ind_y = np.where(end_y < start_y)
    end_y[ind_y] = start_y[ind_y]
    
    intersections = (end_x - start_x) * (end_y - start_y)
    
    # calculate union = A + B - A^B
    unions = bboxes[...,2] * bboxes[...,3] + target[2] * target[3] - intersections
    
    return intersections / unions



def get_anchor_bboxes(img_width, img_height, step_x, step_y, anchor_widths, anchor_heights):
    '''
    Generate the raw bboxes for all the anchors
    
    @params:
    @img_width: scaler, the width of the images
    @img_height: scaler, the height of the images
    @step_x: scaler, step of the anchor centers along x directions
    @step_y: scaler, step of the anchor centers along y directions
    @anchor_widths - list of length nanchors, widths of each anchor
    @anchor_heights - list of length nanchors, heights of each anchor
    
    @returns
    @bboxes: bboxes with the shape of [nanchors, nx, ny, 4], each bbox is [x, y, width, height]. nx and ny are number of bboxes along x and y directions.
    '''
    
    # get list of centers
    center_x = np.arange(step_x//2, img_width, step_x)
    center_y = np.arange(step_y//2, img_height, step_y)
    center_x, center_y = np.meshgrid(center_x, center_y)
    
    bboxes = np.zeros([len(anchor_widths), len(center_x), len(center_y), 4], np.float32)
    
    # each anchor
    for i in range(len(anchor_widths)):
        bboxes[i, ..., 0] = center_x - anchor_widths[i] // 2
        bboxes[i, ..., 1] = center_y - anchor_heights[i] // 2
        bboxes[i, ..., 2] = anchor_widths[i]
        bboxes[i, ..., 3] = anchor_heights[i]
    
    return bboxes



def get_pixel_labels(bboxes, targets, iou_th_low = 0.3, iou_th_high = 0.7):
    '''
    Get the classification and regression labels for each pixel
    
    @params:
    @bboxes: array of shape [nanchors, nx, ny, 4], all the anchor bounding boxes.
    @targets: list of target bounding boxes
    @iou_th_low: below this threshold the pixel is negative
    @iou_th_high: above this threshold the pixel is positive
    
    @returns:
    @cls_labels - classification labels in the shape [nx, ny, nanchors]
    @reg_labels - regression labels in the shape [nx, ny, nanchors * 4]. The arrangement for the least dimension is [x1,y1,w1,h1,x2,y2,w2,h2,...].
    The regression labels are (x - xa) / wa, (y - ya) / ha, log(w / wa), log(h / ha)
    '''
    if len(targets) == 0:
        return (np.zeros([bboxes.shape[1], bboxes.shape[2], bboxes.shape[0]], np.float32), 
    np.zeros([bboxes.shape[1], bboxes.shape[2], bboxes.shape[0] * 4], np.float32))
    
    ious = []
    for target in targets:
        ious.append(IoU(target, bboxes))
    
    ious = np.array(ious)
    inds = np.argmax(ious, 0)
    
    # calculate the label for regression
    reg_labels = np.zeros(list(ious.shape) + [4], np.float32)
    # first calculate the regression labels for all the targets
    for i in range(len(targets)):
        reg_labels[i, ..., 0] = (targets[i][0] - bboxes[..., 0]) / bboxes[..., 2]
        reg_labels[i, ..., 1] = (targets[i][1] - bboxes[..., 1]) / bboxes[..., 3]
        reg_labels[i, ..., 2] = np.log(targets[i][2] / bboxes[..., 2])
        reg_labels[i, ..., 3] = np.log(targets[i][3] / bboxes[..., 3])
        
    # then for each target, select the one with max iou
    inds = np.tile(inds[...,np.newaxis], (1,1,1,reg_labels.shape[-1])).flatten()
    y = reg_labels.reshape(len(targets), -1)
    y = y[inds, np.arange(len(inds))]
    reg_labels = y.reshape(reg_labels.shape[1:])
    # switch the axis and merge the last two axises
    reg_labels = reg_labels.transpose((1,2,0,3))
    reg_labels = reg_labels.reshape((reg_labels.shape[0], reg_labels.shape[1], -1))
    
    # calculate the label for classification
    ious = np.max(ious, 0)
    max_iou = np.max(ious)
    cls_labels = np.zeros_like(ious) - 1          # default -1 means not significant
    cls_labels[ious < iou_th_low] = 0             # < iouThLow means negative samples
    cls_labels[ious >= iou_th_high] = 1           # >= iouThHigh means positive samples
    cls_labels[ious >= max_iou] = 1               # also set the one with max IoU as positive sample
    cls_labels = cls_labels.transpose((1,2,0))    # put the anchors to the channel dimension
    
    return cls_labels, reg_labels


def get_bbox_from_preds(regs, bboxes):
    '''
    Restore the bounding boxes from regression predictions / labels
    
    @params:
    @regs: the regression predictions in the shape of [nx, ny, nanchors * 4]
    @bboxes - the bboxes from anchors, in the shape of [nanchors, nx, ny, 4]
    
    @return: 
    @regs: bounding boxes with reshape [nanchors, nx, ny, 4]
    '''
    regs = np.copy(regs)
    
    regs = regs.reshape((regs.shape[0], regs.shape[1], -1, 4))
    regs = regs.transpose((2, 0, 1, 3))
    
    regs[..., 0] = regs[..., 0] * bboxes[..., 2] + bboxes[..., 0]
    regs[..., 1] = regs[..., 1] * bboxes[..., 3] + bboxes[..., 1]
    regs[..., 2] = np.exp(regs[..., 2]) * bboxes[..., 2]
    regs[..., 3] = np.exp(regs[..., 3]) * bboxes[..., 3]
    
    return regs


def NMS(probs, bboxes, iou_th = 0.7):
    '''
    Non-maximum suppressing
    
    @params:
    @probs: array of [ncandidates], the probability of each prediction
    @bboxes: list of length ncandidates, each element of size [4]. Bounding boxes.
    @iou_th: when two bboxes has iou over iou_th, remove the one with smaller probability
    
    @returns:
    @targets: list of preserved bounding boxes
    @target_probs: list of preserved bounding boxes probability
    '''
    targets = []
    target_probs = []
    candidates = list(bboxes)
    while (len(candidates) > 0):
        ind = np.argmax(probs)
        targets.append(candidates[ind])
        target_probs.append(probs[ind])
        probs = [probs[i] for i in range(len(candidates)) if i != ind]
        candidates = [candidates[i] for i in range(len(candidates)) if i != ind]
        if len(candidates) == 0:
            break
        
        ious = IoU(targets[-1], candidates)
        probs = [probs[i] for i in range(len(ious)) if ious[i] < iou_th]
        candidates = [candidates[i] for i in range(len(ious)) if ious[i] < iou_th]
    
    return targets, target_probs



