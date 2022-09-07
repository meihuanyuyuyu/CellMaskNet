import torch
import numpy as np
from torchvision.utils import save_image,draw_bounding_boxes
from model.maskrcnn import MaskRCNN
from config import arg,anchors_wh
from tools.utils import generate_anchors

from tools.augmentation import My_colorjitter,Random_flip
from tools.dataset import ConicDataset,DataLoader,Subset,collect_fn
from tools.metric import stage1_val
from tqdm import tqdm


stage1_rois = None
def stage1_boxes_hook_for_val(moudle,input,output):
    global stage1_rois
    stage1_rois = input[1]

anchors = generate_anchors(anchors_wh,img_size=256,scale=4).cuda()
net = MaskRCNN(anchors,rpn_pos_threshold=0.7,stage1_mode=False,post_rpn_thresh=0.6).cuda()
net.eval()
net.load_state_dict(torch.load('model_parameters/maskrcnn.pt'))
net.mask_align.register_forward_hook(stage1_boxes_hook_for_val)


imgs = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous()
for i,img in enumerate(imgs):
    img = img.cuda().unsqueeze(0)
    rois,score,out_cls,out_masks = net(img)
    # stage1 boxes
    pic =draw_bounding_boxes((img.squeeze(0) * 255).to(dtype=torch.uint8),stage1_rois[0])
    pic = (pic/255).float().unsqueeze(0)
    save_image(pic,f'figures/msakrcnn_s1_inference/{i}.png')
    
    
