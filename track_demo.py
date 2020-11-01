from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size

from deep_sort_tracker import deep_sort_tracker

import argparse
import time
import cv2
import torch
import torch.backends.cudnn as cudnn


def init_vid_writer(filename, vid_cap):
    fourcc = 'mp4v'
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_writer = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))


def write_trace(bbox_xyxy, ds_tracker):
    watch_data, idx_trace, idx2classid = ds_tracker.watch_data, ds_tracker.idx_trace, ds_tracker.idx2classid
    print('实时统计结果：画面中监控实体数=%d' % (len(bbox_xyxy)))

    txt = "totle = %d " % (len(bbox_xyxy))

    for i, data in enumerate(watch_data):
        stat = {}
        for idx in data:
            class_id = idx2classid[idx]
            if stat.__contains__(class_id):
                stat[class_id] += 1
            else:
                stat[class_id] = 1
        for (class_id, count) in stat.items():
            print('R%d %s流量：%d' % (i, ds_tracker.names[class_id], len(data)))
            txt = txt + "R%d(%s) = %d " % (i,
                                           ds_tracker.names[class_id], len(data))

    return txt


def detect(opt):

    ds_tracker = deep_sort_tracker(opt.config_deepsort)

    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadImages(opt.source)
    if opt.vsave:
        vid_writer = init_vid_writer('data/demo.mp4', dataset.cap)

    bbox_xyxy, identities = None, None
    for frame_idx, (path, img, im0, vid_cap) in enumerate(dataset):

        if frame_idx % opt.skip != 0:
            ds_tracker.draw_boxes(im0, bbox_xyxy, identities)
            im0 = cv2.resize(im0, (im0.shape[1]//2, im0.shape[0]//2))
            cv2.imshow('car flow', im0)
            cv2.imwrite('car_flow.jpg', im0)
            continue

        bbox_xyxy, identities, _ = ds_tracker.predict(im0)

        ds_tracker.draw_boxes(im0, bbox_xyxy, identities)
        ds_tracker.write_trace(frame_idx)
        txt = write_trace(bbox_xyxy, ds_tracker)
        cv2.rectangle(im0, (0, 0), (2000, 100), (0, 0, 0), -1)
        cv2.putText(im0, txt, (50, 50), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 2)

        if opt.vsave:
            vid_writer.write(im0)

        #print(im0.shape, img.shape)
        if opt.show:
            im0 = cv2.resize(im0, (im0.shape[1]//2, im0.shape[0]//2))
            cv2.imshow('car flow', im0)
            if cv2.waitKey(1) == ord('q'):  # q to quit
                raise StopIteration


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')

    parser.add_argument("--config_deepsort", type=str,
                        default="configs/deep_sort.yaml")
    parser.add_argument('--show', action='store_true',
                        help='cv2 show')
    parser.add_argument('--vsave', action='store_true',
                        help='cv2 save')
    parser.add_argument('--skip', type=int, default=1,
                        help='inference size (pixels)')

    args = parser.parse_args()
    args.classes = None

    print(args)

    with torch.no_grad():
        detect(args)
