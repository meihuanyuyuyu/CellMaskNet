from Config.maskrcnn_config import arg,anchors_wh
from model.maskrcnn import MaskRCNN
from tools.utils import convert_prediction_to_numpy
import torch
class MaskRCNNInfer:
    def __init__(self,device,dataset) -> None:
        self.net = MaskRCNN(anchors_wh,rpn_pos_threshold=arg.rpn_pos_thresh,stage2_sample_ratio=arg.stage2_sample_ratio,post_rpn_thresh=arg.post_rpn_pos_thresh,post_detection_iou_thresh=arg.post_detection_nms_thresh,post_detection_score_thresh=arg.post_detection_score_thresh)
        self.net.load_state_dict(torch.load(arg.model_para))
        self.net.eval().to(device=device)
        self.dataset = dataset
    
    def __call__(self,img:torch.Tensor):
        if self.dataset =='conic':
            rois,scores,cls,masks=self.net(img)
            rois,scores,cls,masks = rois[0],scores[0],cls[0],masks[0]
            pred =convert_prediction_to_numpy(rois,cls,masks)
            return pred