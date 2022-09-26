from .maskrcnn import BoxAndClassPredictor, BoxHead, MaskRCNN, proposal_layer,generate_detection_targets,F
from torch import Tensor
from typing import List
import torch


class CascadedRCNN(MaskRCNN):
    def __init__(self, default_anchors: Tensor, stage1_mode: bool, stage2_train_mode: bool, rpn_pos_threshold: float, backbone=..., stage2_sample_ratio=1.5, post_rpn_thresh=0.7, stage2_max_proposal=256, post_detection_score_thresh=0.5, post_detection_iou_thresh=0.2, detections_per_img=500, img_size=...) -> None:
        super().__init__(default_anchors, stage1_mode, stage2_train_mode, rpn_pos_threshold, backbone, stage2_sample_ratio, post_rpn_thresh, stage2_max_proposal, post_detection_score_thresh, post_detection_iou_thresh, detections_per_img, img_size)
        self.post_rpn_thresh = [0.5,0.6,0.7]
        self.box_head = [BoxHead(256,1024) for _ in range(3)]
        self.box_detection = [BoxAndClassPredictor(1024, 7) for _ in range(3)]
        


    def compute_detection_loss(self, batched_rois, detection_box_cls, detection_box_reg, detection_masks, target_boxes, target_cls, target_masks,detection_stage:int=0):
        cls,reg,masks =generate_detection_targets(batched_rois,target_boxes,target_cls,target_masks,out_size=28,iou_thresh=self.post_rpn_thresh[detection_stage])
        if len(cls)!= 0:
            cls = torch.cat(cls,dim=0)
            reg = torch.cat(reg,dim=0)
            masks = torch.cat(masks,dim=0)
            loss = F.cross_entropy(detection_box_cls,cls) + F.cross_entropy(detection_masks[cls!=0],masks[cls!=0]) + F.smooth_l1_loss(detection_box_reg[cls!=0],reg[cls!=0])
        else:
            loss = None
        return {f'detection{detection_stage}_loss':loss}
    
    def forward(self,img:Tensor,target_boxes:List[Tensor]=None,masks:List[Tensor]=None,cls:List[Tensor]=None):
        stride4_feature,stride8_feature = self.backbone(img)
        rpn_logist, rpn_reg =self.rpn(stride4_feature)
        batched_rois,batched_scores =proposal_layer(self.default_anchors,rpn_logist,rpn_reg,is_train=self.training)

        if self.training:
            losses = {}
            if self.stage1_mode and not self.stage2_train_mode:
                rpn_loss =self.compute_rpn_loss(rpn_logist,rpn_reg,target_boxes)
                losses.update(rpn_loss)
                return losses

            if self.stage2_train_mode and not self.stage1_mode:
                for detection in range(len(self.post_rpn_thresh)):
                    batched_rois,batched_scores=self.stage2_proposal_sample(batched_rois,batched_scores,target_boxes,self.post_rpn_thresh[detection])
                    box_feature = self.box_align(stride8_feature,batched_rois)
                    box_feature =self.box_head[detection](box_feature)
                    detection_cls,detection_reg =self.box_detection[detection](box_feature)
                    mask_feature = self.mask_align(stride4_feature,batched_rois)
                    detection_masks =self.mask_detection(mask_feature)

                    detection_loss = self.compute_detection_loss(batched_rois,detection_cls,detection_reg,detection_masks,target_boxes,cls,masks,detection)
                    losses.update(detection_loss)
                return losses 
            
            if self.stage1_mode and self.stage2_train_mode:
                rpn_loss =self.compute_rpn_loss(rpn_logist,rpn_reg,target_boxes)
                losses.update(rpn_loss)
                for detection in range(len(self.post_rpn_thresh)):
                    batched_rois,batched_scores=self.stage2_proposal_sample(batched_rois,batched_scores,target_boxes,self.post_rpn_thresh[detection])
                    box_feature = self.box_align(stride8_feature,batched_rois)
                    box_feature =self.box_head[detection](box_feature)
                    detection_cls,detection_reg =self.box_detection[detection](box_feature)
                    mask_feature = self.mask_align(stride4_feature,batched_rois)
                    detection_masks =self.mask_detection(mask_feature)

                    detection_loss = self.compute_detection_loss(batched_rois,detection_cls,detection_reg,detection_masks,target_boxes,cls,masks,detection)
                    losses.update(detection_loss)
                return losses

        if not self.training:
            for detection in range(len(self.post_rpn_thresh)):
                box_feature = self.box_align(stride8_feature,batched_rois)
                box_feature = self.box_head[detection](box_feature)
                detection_cls,detection_reg = self.box_detection[detection](box_feature)
                mask_feature = self.mask_align(stride4_feature,batched_rois)
                detection_masks = self.mask_detection(mask_feature)
                batched_rois,batched_scores,cls,masks =self.postprocess_detection(detection_cls,detection_reg,detection_masks,batched_rois)
            return batched_rois,batched_scores,cls,masks


