from .maskrcnn import MaskRCNN
import torch.nn as nn
import torch
from typing import List
from torchvision.ops import roi_align

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self,boxes:List[torch.tensor],output_size,mode:str='max',img_size:int=256) -> None:
        super().__init__()
        self.boxes = boxes
        self.output_size = output_size
        self.mode = mode
        self.img_size = img_size

    def forward(self,feature_maps:List[torch.tensor]):
        assert self.mode == 'max' or self.mode =='sum'
        aggregated_feature = [] 
        for feature_map in feature_maps:
            out =roi_align(feature_map,self.boxes,self.output_size,spatial_scale=feature_map.shape[-1]//self.img_size,aligned=True)
            aggregated_feature.append(out)
        if self.mode == 'max':
            return torch.max(torch.stack(aggregated_feature,dim=0),dim=0).values
        if self.mode=='sum':
            return torch.sum(torch.stack(aggregated_feature,dim=0),dim=0)
        
class CellMaskNet(MaskRCNN):
    def __init__(self, default_anchors: torch.Tensor, stage1_mode: bool, stage2_train_mode: bool, rpn_pos_threshold: float, backbone=..., stage2_sample_ratio=1.5, post_rpn_thresh=0.7, stage2_max_proposal=256, post_detection_score_thresh=0.5, post_detection_iou_thresh=0.2, detections_per_img=500, img_size=...) -> None:
        super().__init__(default_anchors, stage1_mode, stage2_train_mode, rpn_pos_threshold, backbone, stage2_sample_ratio, post_rpn_thresh, stage2_max_proposal, post_detection_score_thresh, post_detection_iou_thresh, detections_per_img, img_size)
        pass
