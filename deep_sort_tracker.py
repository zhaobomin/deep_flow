from configs.parser import get_config
from deep_sort.deep_sort import DeepSort
from yolov5.yolo_net import yolo_net
import torch
import cv2
import time


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        x1_min = min(box1[0], box2[0])
        x2_max = max(box1[2], box2[2])
        y1_min = min(box1[1], box2[1])
        y2_max = max(box1[3], box2[3])
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    else:
        w1, h1 = box1[2], box1[3]
        w2, h2 = box2[2], box2[3]
        x1_min = min(box1[0] - w1 / 2.0, box2[0] - w2 / 2.0)
        x2_max = max(box1[0] + w1 / 2.0, box2[0] + w2 / 2.0)
        y1_min = min(box1[1] - h1 / 2.0, box2[1] - h2 / 2.0)
        y2_max = max(box1[1] + h1 / 2.0, box2[1] + h2 / 2.0)

    w_union = x2_max - x1_min
    h_union = y2_max - y1_min
    w_cross = w1 + w2 - w_union
    h_cross = h1 + h2 - h_union
    carea = 0
    if w_cross <= 0 or h_cross <= 0:
        return 0.0

    area1 = w1 * h1
    area2 = w2 * h2
    carea = w_cross * h_cross
    uarea = area1 + area2 - carea
    return float(carea / uarea)


class deep_sort_tracker():
    def __init__(self, config_deepsort, classes_filter=[2, 5, 7, 0]):
        cfg = get_config()
        cfg.merge_from_file(config_deepsort)
        self.deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
                                 max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                                 nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                 max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                 use_cuda=True)
        self.yolo = yolo_net(cfg.YOLO.YOLO_WEIGHTS, cfg.YOLO.IMG_SIZE)
        self.conf_thres = cfg.YOLO.CONF_THRES
        self.iou_thres = cfg.YOLO.IOU_THRES

        self.names = self.yolo.names

        self.idx2classid = {}
        self.classes_filter = classes_filter
        self.last_ids = []
        self.watch_box = cfg.WATCH_BOX  # watch_box
        self.watch_data = []
        self.idx_trace = {}

        self.trace_writer = None

        for i in range(len(self.watch_box)):
            self.watch_data.append(set())

    def _match_box(self, bbox, identities, orig_bbox, clss):
        # clsid = -1
        for i, idx in enumerate(identities):
            self.idx2classid[idx] = clss[i]
            for j, watch_box in enumerate(self.watch_box):
                if bbox_iou(watch_box, bbox[i]) > 0:
                    self.watch_data[j].add(idx)
                    if self.idx_trace.__contains__(idx):
                        if self.idx_trace[idx][-1] != j:
                            self.idx_trace[idx].append(j)
                    else:
                        self.idx_trace[idx] = [j]

    def predict(self, img):
        pred = self.yolo.predict(img, self.conf_thres, self.iou_thres)
        bbox_xyxy, identities, clss = self._update(pred, img)
        return bbox_xyxy, identities, clss

    def write_trace(self, frame_idx=0):
        if self.trace_writer == None:
            self.trace_writer = open(
                'data/tracelog_%d.log' % (int(time.time())), 'w')
            self.trace_writer.write("NEW_TRACE:%d\n" % (int(time.time())))

        log = 'TRACE:%d' % (frame_idx)
        for (idx, trace) in self.idx_trace.items():
            trace_txt = ','.join(str(t) for t in trace)
            log += '\t%d:%s' % (idx, trace_txt)
        log += '\n'
        self.trace_writer.write(log)

    def draw_boxes(self, img, bbox, identities):

        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]

            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = compute_color_for_labels(id)
            label = ''

            clsid = self.idx2classid[id]

            clsname = self.names[clsid]
            label = '%s id=%d' % (clsname, id)

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)

        for i, w_box in enumerate(self.watch_box):
            cv2.rectangle(img, (w_box[0], w_box[1]),
                          (w_box[2], w_box[3]), (0, 255, 0), 4)

            cv2.putText(img, "R%d" % (i),
                        (w_box[0], w_box[1]), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)

        return img

    def _update(self, bboxes_pred, im0):

        det = bboxes_pred
        bbox_xyxy = []
        identities = []
        if det is not None and len(det) > 0:
            bbox_xywh = []
            confs = []
            clss = []
            # Adapt detections to deep sort input format
            for *xyxy, conf, clsid in det:
                if clsid not in self.classes_filter:
                    continue
                x_c, y_c, bbox_w, bbox_h = bbox_rel(*xyxy)
                obj = [x_c, y_c, bbox_w, bbox_h]
                bbox_xywh.append(obj)
                confs.append([conf.item()])
                clss.append(clsid)

            if len(bbox_xywh) > 0:
                xywhs = torch.Tensor(bbox_xywh)
                confss = torch.Tensor(confs)
                # print('==', bbox_xywh)
                # Pass detections to deepsort
                outputs = self.deepsort.update(xywhs, confss, im0, clss)

                # draw boxes for visualization
                if len(outputs) > 0:
                    # print(len(outputs[0]))
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -2]
                    clss = outputs[:, -1]
                    self._match_box(bbox_xyxy, identities, det, clss)

        return bbox_xyxy, identities, clss
