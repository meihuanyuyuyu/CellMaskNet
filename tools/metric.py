from cmath import isnan
from operator import index
from typing import List
from torchvision.utils import draw_bounding_boxes,make_grid,save_image
from torchvision.ops import box_iou
import torch


def proposal_targetboxes_miou(proposal: List[torch.Tensor], target_boxes: List[torch.Tensor]):
    max_ious = torch.zeros(0,device='cuda') 
    for proposal,target_boxes in zip(proposal,target_boxes):
        iou = box_iou(proposal, target_boxes)
        if len(target_boxes) ==0:
            continue
        max_iou = torch.max(iou, dim=-1).values
        max_ious = torch.cat([max_ious,max_iou],dim=0)
    return max_ious.mean().item()

def stage1_val(boxes:List[torch.Tensor],preds:List[torch.Tensor],imgs:torch.Tensor,fp):
    r'根据一阶段roi和boxes,统计验证集上miou并可视化保存效果图片'
    imgs = (imgs * 255).to(torch.uint8)
    pic = torch.zeros(0,3,256,256,device='cuda')
    for img,box,pred in zip(imgs,boxes,preds):
        true = (draw_bounding_boxes(img, box) / 255).cuda().unsqueeze(0)
        pred_boxes = (draw_bounding_boxes(img, pred) / 255).cuda().unsqueeze(0)
        pic =torch.cat([pic,true,pred_boxes],dim=0)
    pic=make_grid(pic,2)
    save_image(pic,fp=fp)
    return proposal_targetboxes_miou(preds,boxes)

def proposal_stage2_metric(proposal: List[torch.Tensor],pred_masks,pred_clses, target_boxes: List[torch.Tensor], target_masks: List[torch.Tensor], target_cls: List[torch.Tensor]):
    '每张图片boxes平均iou,正例类别标签中正确的占比,mask 平均iou'
    max_boxes_ious = torch.zeros(0,device='cuda')
    max_masks_ious = torch.zeros(0,device='cuda')
    all_cls = torch.zeros(0,device='cuda')
    for roi,pred_mask,pred_cls,boxes,masks,cls in zip(proposal,pred_masks,pred_clses,target_boxes,target_masks,target_cls):
        if len(boxes) ==0:
            continue
        iou = box_iou(roi, boxes)
        max_iou,index = torch.max(iou, dim=-1)
        masks = masks[index]
        cls = cls[index]
        max_masks_iou = jaccard(pred_mask,masks)
        cls:torch.Tensor = (pred_cls[pred_cls.bool()]==cls[cls.bool()]).sum()/len(cls[cls.bool()])
        
        if cls.isnan().any() or max_masks_iou.isnan().any() or max_iou.isnan().any():
            continue
        else:
            max_masks_ious = torch.cat([max_masks_ious,torch.tensor([max_masks_iou],device=max_boxes_ious.device)],dim=0)
            max_boxes_ious = torch.cat([max_boxes_ious,max_iou],dim=0)
            all_cls = torch.cat([all_cls,torch.tensor([cls.item()],device=all_cls.device)],dim=0)
    return max_boxes_ious.mean().item(),max_masks_ious.mean().item(),all_cls.mean().item()

def jaccard(masks:torch.Tensor,target_masks:torch.Tensor):
    TP = ((masks==1) & (target_masks==1)).sum()
    FP_FN = torch.ne(masks,target_masks).sum()
    return (TP/(TP+FP_FN)).item()

