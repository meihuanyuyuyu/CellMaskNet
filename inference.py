import torch
import numpy as np
from model.maskrcnn import MaskRCNN
from Config.maskrcnn_config import arg,anchors_wh
from tools.utils import generate_anchors, rois2img,convert_prediction_to_numpy
import os
from tqdm import tqdm

@torch.no_grad()
def get_final_metric():
    anchors = generate_anchors(anchors_wh,img_size=256,scale=4).cuda()
    net = MaskRCNN(anchors,rpn_pos_threshold=arg.rpn_pos_thresh,stage1_mode=False,stage2_train_mode=False,post_rpn_thresh=arg.post_rpn_pos_thresh,).cuda()
    net.eval()
    net.load_state_dict(torch.load(arg.model_para))

    imgs = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous().cuda()
    gts = np.load('project_conic/CoNIC_Challenge/labels.npy')
    test_index = torch.load('train_test_indexes/splitted_indexes.pt')['test']

    preds = torch.zeros(0, 2, 256, 256, dtype=torch.long)
    _gts = np.zeros((0, 256, 256, 2), dtype=np.uint16)
    for i,index in enumerate(tqdm(test_index)):
        img = imgs[index:index+1]
        gt =gts[index:index+1]
        rois,scores,cls,masks =net(img)
        rois,scores,cls,masks = rois[0],scores[0],cls[0],masks[0]
        pred =convert_prediction_to_numpy(rois,cls,masks)
        preds = torch.cat([preds,pred],dim=0)
        _gts = np.concatenate((_gts,gt),axis=0)
    preds = preds.permute(0,2,3,1).numpy().astype(np.uint16)
    np.save(os.path.join(arg.numpy_dir,arg.__class__.__name__),preds)
    np.save(os.path.join(arg.numpy_dir,arg.__class__.__name__+'_gt'),_gts)


print('starting inferencing!')
get_final_metric()

    
    
