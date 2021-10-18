# --------------------------------------------------------
# dota_evaluation_task1
# Licensed under The MIT License [see LICENSE for details]
# Written by Jian Ding, based on code from Bharath Hariharan
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import xml.etree.ElementTree as ET
import os
#import cPickle
import numpy as np
from mmdetection.polyiou import polyiou
from functools import partial
import glob

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects

# parse_gt('/content/EfficientDet/DOTA_split_600_150/val/labelTxt/P0003_0000_0000.txt') 




def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def voc_eval_dota(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5,
             use_07_metric=False):
    """rec, prec, ap = voc_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh],
                                [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
        (default False)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file

    # first load gt
    #if not os.path.isdir(cachedir):
     #   os.mkdir(cachedir)
    #cachefile = os.path.join(cachedir, 'annots.pkl')
    # read list of images
    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    #print('imagenames: ', imagenames)
    #if not os.path.isfile(cachefile):
        # load annots
    recs = {}
    for i, imagename in enumerate(imagenames):
        # print('parse_files name: ', annopath.format(imagename))
        recs[imagename] = parse_gt(annopath.format(imagename))
        # recs[imagename] = parse_gt(annopath.format(osp.basename(imagename).split('.png')[0]))
    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    #print('check confidence: ', confidence)

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)

    #print('check sorted_scores: ', sorted_scores)
    #print('check sorted_ind: ', sorted_ind)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    #print('check imge_ids: ', image_ids)
    #print('imge_ids len:', len(image_ids))
    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT

        if BBGT.size > 0:
            # compute overlaps
            # intersection

            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            BBGT_keep_index = np.where(overlaps > 0)[0]
            # pdb.set_trace()
            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):

                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)

                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)
                # pdb.set_trace()
                jmax = BBGT_keep_index[jmax]
        if ovmax > ovthresh:
            if not R['difficult'][jmax]:
                if not R['det'][jmax]:
                    tp[d] = 1.
                    R['det'][jmax] = 1
                else:
                    fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall

    # print('check fp:', fp)
    # print('check tp', tp)


    # print('npos num:', npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return rec, prec, ap
# from help_utils import tools
wordname_15 = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
def write_voc_results_file(all_boxes, test_imgid_list, det_save_dir):
  '''
  :param all_boxes: is a list. each item reprensent the detections of a img.
  the detections is a array. shape is [-1, 7]. [category, score, x, y, w, h, theta]
  Note that: if none detections in this img. that the detetions is : []
  :param test_imgid_list:
  :param det_save_path:
  :return:
  '''
  for cls_id,cls  in enumerate(wordname_15):
    if cls == 'back_ground':
      continue
    print("Writing {} VOC resutls file".format(cls))

    # tools.mkdir(det_save_dir)
    det_save_path = os.path.join(det_save_dir, "Task1_"+cls+".txt")
    with open(det_save_path, 'wt') as f:
      for index, img_name in enumerate(test_imgid_list):
        this_img_detections = all_boxes[index]

        this_cls_detections = this_img_detections[this_img_detections[:, 0] == cls_id]
        if this_cls_detections.shape[0] == 0:
          continue # this cls has none detections in this img
        for a_det in this_cls_detections:
          f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                  format( os. path. splitext(os.path.basename(img_name))[0], a_det[1],
                         a_det[2], a_det[3],
                         a_det[4], a_det[5],
                         a_det[6],a_det[7],
                         a_det[8],a_det[9]))  # that is [img_name, score, x, y, w, h, theta]

def poly_nms(dets,scores, iou_threshold):
      # scores = dets[:, 8]
      polys = []
      areas = []
      for i in range(len(dets)):
          tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                              dets[i][2], dets[i][3],
                                              dets[i][4], dets[i][5],
                                              dets[i][6], dets[i][7]])
          polys.append(tm_polygon)
      order = scores.argsort()[::-1]

      keep = []
      while order.size > 0:
          ovr = []
          i = order[0]
          keep.append(i)
          for j in range(order.size - 1):
              iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
              ovr.append(iou)
          ovr = np.array(ovr)
          inds = np.where(ovr <= iou_threshold)[0]
          order = order[inds + 1]
      return keep

def py_cpu_nms_poly_fast(dets,scores, thresh):
    try:
        obbs = dets[:, 0:-1]
    except:
        print('fail index')
        pdb.set_trace()
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    # scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep

def poly_nms_np(quads,scores,lables):
    quadrangles = np.reshape(quads,(-1,8))
    idx = py_cpu_nms_poly_fast(quadrangles.astype(np.float64),scores,.1)
    # print(idx)
    quads = quads[idx]
    scores = scores[idx]
    lables = lables[idx]
    return quads, scores, lables

def rotate(ct,angle):
      # rotation_amount_rad = angle * np.pi / 180.0
      # c1=(ct[:,0].mean(),ct[:,1].mean())
      # a=np.cos(rotation_amount_rad)
      # b=np.sin(rotation_amount_rad)
    #   M=np.array([[  a , b,  (1-a)*c1[0] -  b*c1[1]    ] ,
    #               [  -b , a,   b*c1[0]    + (1-a)*c1[1] ]])
      M = cv2.getRotationMatrix2D(tuple(ct.mean(axis=0)),angle, 1.0)
      # print(M)
      ones = np.ones(shape=(len(ct), 1))

      points_ones = np.hstack([ct, ones])

      # transform points
      return  (M.dot(points_ones.T).T).astype(np.float32)

def result_to_dets(result):
    d = [ np.concatenate((r, np.repeat(i,len(r))[..., np.newaxis] ),axis=1)
                            for i,r in enumerate( result ) if len(r)>0]
    # print(len(d))                        
    if len(d):                        
      dets = np.concatenate(d)
      angles = dets[...,0]
      quads_off = dets[...,1:5]
      scores = dets[...,5]
      lables = dets[...,6]

      quads = np.zeros((quads_off.shape[0] , 8))
      quads[:, 0] = quads_off[:, 0] 
      quads[:, 1] = quads_off[:, 1]
      quads[:, 2] = quads_off[:, 2]
      quads[:, 3] = quads_off[:, 1] 
      quads[:, 4] = quads_off[:, 2] 
      quads[:, 5] = quads_off[:, 3]
      quads[:, 6] = quads_off[:, 0]
      quads[:, 7] = quads_off[:, 3] 
      quads = np.reshape(quads,(-1,4,2))
      quads = np.array([rotate(q,-r)  for q,r in zip(quads,angles)])
      return quads, scores, lables
    else:
      quads = np.zeros((1,8))
      scores = np.zeros((1))
      lables = np.zeros((1))
      return quads, scores, lables
