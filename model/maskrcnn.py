from types import MethodType
from typing import List
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50,resnet101,resnext101_64x4d,resnext50_32x4d,ResNeXt50_32X4D_Weights,ResNet50_Weights
from torchvision.ops import RoIAlign,box_iou,remove_small_boxes,nms,clip_boxes_to_image
from tools.utils import proposal_layer,generate_rpn_targets,balanced_pos_neg_sample,generate_detection_targets,apply_box_delta, remove_big_boxes,focal_loss



class conv_bn_activ(nn.Module):
    def __init__(self,in_c,out_c,kernel_size,stride,padding,active_f:nn.Module=nn.LeakyReLU) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c,out_c,kernel_size,stride=stride,padding=padding),
            nn.BatchNorm2d(out_c),
            active_f(inplace = True),
        )

    def forward(self,x):
        return self.conv(x)

class FPN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.x4_conv = nn.Conv2d(2048,512,1,1) #/32
        self.x3_conv = nn.Conv2d(1024,512,1,1) #16
        self.x2_conv = nn.Conv2d(512,512,1,1) #8
        self.x1_conv = nn.Conv2d(256,512,1,1) #/4

        self.out_2 = nn.Conv2d(512,256,3,1,1)
        self.out_1 = nn.Conv2d(512,256,3,1,1)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self,c1,c2,c3,c4):
        x4 = self.x4_conv(c4)
        x3 = self.x3_conv(c3)+self.up(x4)
        x2 = self.x2_conv(c2)+self.up(x3)
        x1 = self.x1_conv(c1)+self.up(x2)
        return self.out_1(x1),self.out_2(x2)

class RPN(nn.Module):
    '对resnet某一层进行预测'
    def __init__(self,depth,num_anchors) -> None:
        super().__init__()
        self.depth = depth
        self.k = num_anchors
        self.base_conv = conv_bn_activ(self.depth,256,3,1,1)
        self.conv_cls = conv_bn_activ(256,self.k,1,1,0)
        self.conv_reg = conv_bn_activ(256,4*self.k,1,1,0)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self,x):
        x = self.base_conv(x)
        x_cls = self.conv_cls(x)
        x_reg = self.conv_reg(x)
        return self.sigmoid(x_cls),x_reg

def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)

    x1 = self.layer1(x)
    x2 = self.layer2(x1)
    x3 = self.layer3(x2)
    x4 = self.layer4(x3)
    return x1,x2,x3,x4

'''class Resnext101(nn.Module):
    def __init__(self,pretrain) -> None:
        super().__init__()
        if pretrain:
            self.backbone = res'''



class Resnet50FPN(nn.Module):
    def __init__(self,pretrain) -> None:
        super().__init__()
        if pretrain:
            self.bcakbone = resnet50(pretrained=True)
            del self.bcakbone.fc
        else:
            self.bcakbone = resnet50()
            del self.bcakbone.fc
        self.bcakbone.forward = MethodType(forward,self.bcakbone)
        self.fpn = FPN()
    
    def forward(self,x):
        x1,x2,x3,x4=self.bcakbone(x)
        stride4_feature,stride8_featrue =self.fpn(x1,x2,x3,x4)
        return stride4_feature,stride8_featrue

class BoxHead(nn.Module):
    def __init__(self, in_channels, representation_size):
        super().__init__()
        self.fc6 = nn.Linear(in_channels*7**2, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)

    def forward(self, x):
        x = x.flatten(start_dim=1) # n,1024
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return x

class BoxAndClassPredictor(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels,  4)


    def forward(self, x):
        if x.dim() == 4:
            assert list(x.shape[2:]) == [1, 1]
        x = x.flatten(start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores , bbox_deltas

class MaskRCNN(nn.Module):
    def __init__(self,default_anchors:Tensor,stage1_mode:bool,stage2_train_mode:bool,rpn_pos_threshold:float,backbone = Resnet50FPN(False),stage2_sample_ratio=1.5,post_rpn_thresh=0.7,stage2_max_proposal=256,post_detection_score_thresh = 0.5,post_detection_iou_thresh = 0.2,detections_per_img = 500,img_size = [256,256],loss_cls_weight:float=1.0,class_weight=None) -> None:
        super().__init__()
        self.stage1_mode = stage1_mode
        self.default_anchors = default_anchors
        self.rpn_pos_thresh = rpn_pos_threshold
        self.backbone = backbone
        self.rpn = RPN(256,15)
        self.stage2_train_mode = stage2_train_mode
        self.stage2_sample_ratio = stage2_sample_ratio
        self.post_rpn_thresh = post_rpn_thresh
        self.stage2_max_proposal = stage2_max_proposal
        self.box_align = RoIAlign(7,0.125,-1,aligned=True)
        self.mask_align = RoIAlign(28,0.25,-1,aligned=True)
        self.box_head = BoxHead(256,1024)
        self.box_detection = BoxAndClassPredictor(1024,7)
        self.mask_detection = nn.Sequential(
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),            
            nn.Conv2d(256,128,3,1,1),
            nn.ReLU(True),
            nn.Conv2d(128,64,1,1),
            nn.ReLU(True),
            nn.Conv2d(64,2,1,1),
            )
        self.post_detection_score_thresh = post_detection_score_thresh
        self.post_detection_iou_thresh = post_detection_iou_thresh
        self.detections_per_img = detections_per_img
        self.img_size = img_size
        self.loss_cls_weight = loss_cls_weight
        self.class_weight = class_weight
        

    def stage2_proposal_sample(self,batched_rois,batched_score,target_boxes,post_rpn_thresh:float=0.7):
        r'将第一阶段输出进行正负例平衡采样，并选取不超过256的数量来作为二阶段网络训练。'
        for _ in range(len(batched_rois)):
            if len(batched_rois[_])==0 or target_boxes[_].shape[0]==0:
                continue
            rois_per_img = batched_rois[_]
            score_per_img = batched_score[_]
            target_boxes_per_img = target_boxes[_]
            iou = box_iou(rois_per_img,target_boxes_per_img)
            max_iou = torch.max(iou,dim=-1).values
            pos = max_iou >= post_rpn_thresh
            neg = balanced_pos_neg_sample(pos,max_iou < post_rpn_thresh,sample_ratio=self.stage2_sample_ratio)
            keep = torch.logical_or(pos,neg)
            batched_rois[_] = rois_per_img[keep]
            batched_score[_] = score_per_img[keep]
            if len(batched_rois[_]) > self.stage2_max_proposal:
                batched_rois[_] = batched_rois[_][:self.stage2_max_proposal]
                batched_score[_] =  batched_score[_][:self.stage2_max_proposal]
        return batched_rois,batched_score

    def compute_rpn_loss(self,rpn_logist,rpn_reg,target_boxes):
        rpn_logist_target,rpn_reg_target = generate_rpn_targets(self.default_anchors,target_boxes,self.post_rpn_thresh)
        rpn_reg = rpn_reg.view(rpn_logist.shape[0],4,15,rpn_logist.shape[2],rpn_logist.shape[3]).permute(0,2,3,4,1)
        loss_logist = F.binary_cross_entropy(rpn_logist[rpn_logist_target>=0],rpn_logist_target[rpn_logist_target>=0])
        loss_reg = F.smooth_l1_loss(rpn_reg[rpn_logist_target==1],rpn_reg_target[rpn_logist_target==1])
        return {'rpn_loss':loss_reg+loss_logist}
    
    def compute_detection_loss(self,batched_rois,detection_box_cls,detection_box_reg,detection_masks,target_boxes,target_cls,target_masks):
        cls,reg,masks =generate_detection_targets(batched_rois,target_boxes,target_cls,target_masks,out_size=28,iou_thresh=self.post_rpn_thresh)
        if len(cls)!= 0:
            cls = torch.cat(cls,dim=0)
            print('二阶段标签',torch.bincount(cls).tolist())
            reg = torch.cat(reg,dim=0)
            masks = torch.cat(masks,dim=0)
            loss = self.loss_cls_weight*focal_loss(detection_box_cls,cls,gama=2,weight=self.class_weight) + F.cross_entropy(detection_masks[cls!=0],masks[cls!=0]) + F.smooth_l1_loss(detection_box_reg[cls!=0],reg[cls!=0])
        else:
            loss = None
        return {'detection_loss':loss}

    def postprocess_detection(self,detection_cls:Tensor,detection_reg:Tensor,detection_masks:Tensor,proposal:List[Tensor]):
        r'二阶段后处理'
        scores,detection_cls = torch.max(detection_cls,dim=1)
        detection_masks = torch.argmax(detection_masks,dim=1)
        num_boxes_per_img = [boxes.shape[0] for boxes in proposal]
        detection_cls = detection_cls.split(num_boxes_per_img,0)
        detection_reg = detection_reg.split(num_boxes_per_img,0)
        detection_masks = detection_masks.split(num_boxes_per_img,0)
        scores = scores.split(num_boxes_per_img,0)

        proposal_list = []
        score_list =[]
        cls_list = []
        masks_list = []
        for boxes,score,cls,reg,masks in zip(proposal,scores,detection_cls,detection_reg,detection_masks):
            #remove background proposals
            cls_bool = cls.bool()
            boxes,score,cls,reg,masks = boxes[cls_bool],score[cls_bool],cls[cls_bool],reg[cls_bool],masks[cls_bool]
            
            # remove low scoring boxes
            score_bool = score >= self.post_detection_score_thresh
            boxes,score,cls,reg,masks = boxes[score_bool],score[score_bool],cls[score_bool],reg[score_bool],masks[score_bool]

            boxes = apply_box_delta(boxes,reg)

            boxes = clip_boxes_to_image(boxes,self.img_size)
            keep = remove_big_boxes(boxes,70)
            boxes,score,cls,masks = boxes[keep],score[keep],cls[keep],masks[keep]
            keep = remove_small_boxes(boxes,1.5)
            boxes,score,cls,masks = boxes[keep],score[keep],cls[keep],masks[keep]


            #nms
            keep = nms(boxes,score,self.post_detection_iou_thresh)
            keep = keep[:self.detections_per_img]
            boxes,score,cls,masks = boxes[keep],score[keep],cls[keep],masks[keep]

            proposal_list.append(boxes)
            score_list.append(score)
            cls_list.append(cls)
            masks_list.append(masks)
        
        return proposal_list,score_list,cls_list,masks_list

    def forward(self,img:Tensor,target_boxes:List[Tensor]=None,masks:List[Tensor]=None,cls:List[Tensor]=None):
        stride4_feature,stride8_feature = self.backbone(img)
        rpn_logist, rpn_reg =self.rpn(stride4_feature)
        batched_rois,batched_scores =proposal_layer(self.default_anchors,rpn_logist,rpn_reg,is_train=self.training)

        if self.training:
            losses = {}
            # 二阶段训练
            if self.stage2_train_mode and not self.stage1_mode:
                batched_rois,batched_scores= self.stage2_proposal_sample(batched_rois,batched_scores,target_boxes,self.post_rpn_thresh) #list[tensor]
                box_feature = self.box_align(stride8_feature,batched_rois)
                box_feature =self.box_head(box_feature)
                detection_cls,detection_reg =self.box_detection(box_feature)
                mask_feature = self.mask_align(stride4_feature,batched_rois)
                detection_masks =self.mask_detection(mask_feature)


                detection_loss = self.compute_detection_loss(batched_rois,detection_cls,detection_reg,detection_masks,target_boxes,cls,masks)
                losses.update(detection_loss)
                return losses

            # 一阶段训练
            if self.stage1_mode and not self.stage2_train_mode:
                rpn_loss =self.compute_rpn_loss(rpn_logist,rpn_reg,target_boxes)
                losses.update(rpn_loss)
                return losses
            # 联合训练
            if self.stage1_mode and self.stage2_train_mode:
                rpn_loss =self.compute_rpn_loss(rpn_logist,rpn_reg,target_boxes)
                losses.update(rpn_loss)
                batched_rois,batched_scores = self.stage2_proposal_sample(batched_rois,batched_scores,target_boxes,self.post_rpn_thresh) #list[tensor]
                box_feature = self.box_align(stride8_feature,batched_rois)
                box_feature =self.box_head(box_feature)
                detection_cls,detection_reg =self.box_detection(box_feature)
                mask_feature = self.mask_align(stride4_feature,batched_rois)
                detection_masks =self.mask_detection(mask_feature)
                #预测是否与生成标签等长
                detection_loss = self.compute_detection_loss(batched_rois,detection_cls,detection_reg,detection_masks,target_boxes,cls,masks)
                losses.update(detection_loss)
                return losses
        
        if not self.training:
            box_feature = self.box_align(stride8_feature,batched_rois)
            box_feature = self.box_head(box_feature)
            detection_cls,detection_reg = self.box_detection(box_feature)
            mask_feature = self.mask_align(stride4_feature,batched_rois)
            detection_masks = self.mask_detection(mask_feature)
            batched_rois,batched_scores,cls,masks =self.postprocess_detection(detection_cls,detection_reg,detection_masks,batched_rois)

            return batched_rois,batched_scores,cls,masks
            
            
            
            



        