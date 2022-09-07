import torch
from torchvision.ops import masks_to_boxes,remove_small_boxes,box_convert,box_iou
from tools.utils import remove_big_boxes
from torch.nn.functional import one_hot
from tqdm import tqdm
import numpy as np

def k_means_wh(num_anchors: int):
    labels = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy')[..., 0].astype(np.uint8)).long()
    # 得到边界框
    boxes_all = torch.zeros(0, 4)
    bar = tqdm(labels)
    for label in bar:
        if label.max() == 0:
            continue
        boxes = masks_to_boxes(one_hot(label, num_classes=label.max() + 1)[..., 1:].permute(2, 0, 1).contiguous())
        indexes = remove_small_boxes(boxes, 0.5)
        boxes = boxes[indexes]
        # 去掉标注错误大边界框
        boxes = remove_big_boxes(boxes, 70.)
        boxes_all = torch.cat([boxes_all, boxes], dim=0)
    # 都归到相同中心位置
    boxes_wh = box_convert(boxes_all, 'xyxy', 'cxcywh')
    boxes_wh[..., :2] = 0
    anchors = torch.stack([torch.zeros(num_anchors), torch.zeros(num_anchors), torch.linspace(1, 20, num_anchors), torch.linspace(1, 20, num_anchors)], dim=-1)
    refined_boxes_all = box_convert(boxes_wh, 'cxcywh', 'xyxy')
    print(anchors.shape)

    for _ in range(50):
        # n,15
        anchors_xyxy = box_convert(anchors, 'cxcywh', 'xyxy')
        d_ious = 1 - box_iou(refined_boxes_all, anchors_xyxy)
        # n
        indexes = torch.argmin(d_ious, dim=1)
        for _ in range(num_anchors):
            index = indexes == _
            wh = boxes_wh[index][..., 2:].mean(dim=0)
            anchors[_, 2:] = wh
    return anchors[..., 2:].clone()