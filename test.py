from model.maskrcnn import MaskRCNN
from tools.dataset import ConicDataset
import torch
from torch.nn.functional import interpolate
from config import anchors_wh
from tools.utils import generate_detection_targets,generate_anchors,proposal_layer,anchors_to_max_boxes_delta,apply_box_delta, rois2img
from torchvision.ops import roi_align,remove_small_boxes,clip_boxes_to_image
from torchvision.utils import draw_bounding_boxes,save_image
from torchvision.transforms.functional import pil_to_tensor

def stage1_boxes_hook_for_val(moudle,input,output):
    global stage1_rois
    stage1_rois = input[1]


idxs = [1819,210,87,29,28,26,94]
stage1_rois = None

anchors = generate_anchors(anchors_wh,256,4).cuda()
net = MaskRCNN(anchors,stage1_mode=False,stage2_train_mode=True,rpn_pos_threshold=0.7,post_rpn_thresh=0.7).cuda()
net.load_state_dict(torch.load('model_parameters/maskrcnn.pt'))


net.box_align.register_forward_hook(stage1_boxes_hook_for_val)


net.eval()
color = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32, device='cuda')
data = ConicDataset()
for i,idx in enumerate(idxs):

    result = torch.zeros(1,256,256,dtype=torch.float,device=color.device)
    result2 = torch.zeros(1,256,256,dtype=torch.float,device=color.device)
    boxes,masks,cls=data.generate_targets_boxes(data.labels[idx].cuda(),28)


    # labels
    pic0 = draw_bounding_boxes((data.imgs[idx]*255).to(device='cuda',dtype=torch.uint8),boxes)/255




    rois,scores,out_cls,out_masks=net(data.imgs[idx:idx+1].cuda())
    print('一阶段预测结果',type(stage1_rois))
    
    cls,reg,masks= generate_detection_targets(stage1_rois,[boxes],[cls],[masks],28,0.5)

    cls,reg,masks = cls[0],reg[0],masks[0]
    pic1 =  draw_bounding_boxes((data.imgs[idx]*255).to(device='cuda',dtype=torch.uint8),stage1_rois[0])/255
    boxes = apply_box_delta(stage1_rois[0],reg)[cls!=0]
    pic2 = draw_bounding_boxes((data.imgs[idx]*255).to(device='cuda',dtype=torch.uint8),boxes)/255


    index = remove_small_boxes(boxes,1.5)
    masks = masks[cls!=0][index]
    cls = cls[cls!=0][index]
    boxes = boxes[index]
    boxes = boxes.long()
    boxes[:,2:] +=1
    boxes= clip_boxes_to_image(boxes,[256,256])

    for _ in range(len(boxes)):
        w = (boxes[_,2]-boxes[_,0]).item()
        h = (boxes[_,3]-boxes[_,1]).item()
        mask = interpolate(masks[_:_+1,None].float(),size=[h,w],mode='bilinear').squeeze(0).squeeze(0)
        result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w] = result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w] + (mask.round().long() & (result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w]==False)) * cls[_].item()

    pic3 = color[result.long()].permute(0,3,1,2)
    pic3 = torch.cat([pic3,color[data.labels[idx][1:2]].permute(0,3,1,2)],dim=0).cpu()
    pic = torch.cat([pic0[None],pic1[None],pic2[None],pic3])
    save_image(pic,f'test/test2/{i}.png')


'''
idxs = [1819,210,87,29,28,26,94]
color = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32, device='cuda')
data = ConicDataset()
for i,idx in enumerate(idxs):
    boxes,masks,cls=data.generate_targets_boxes(data.labels[idx],28)

    pic0 = draw_bounding_boxes((data.imgs[idx]*255).to(device='cuda',dtype=torch.uint8),boxes)/255
    result = torch.zeros(1,256,256,dtype=torch.float,device=boxes[0].device)
    boxes = boxes.long()  
    boxes[:,2:] += 1

    boxes=clip_boxes_to_image(boxes,[256,256])
    index = remove_small_boxes(boxes,1.5)
    boxes =boxes[index]

    for _ in range(len(boxes)):
        w = (boxes[_,2]-boxes[_,0]).item()
        h = (boxes[_,3]-boxes[_,1]).item()
        mask = interpolate(masks[_:_+1,None].float(),size=[h,w],mode='bilinear',align_corners=True).squeeze(0).squeeze(0)

        result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w] = result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w] + (mask.round().long() & (result[0,boxes[_,1]:boxes[_,1]+h,boxes[_,0]:boxes[_,0]+w]==False))* cls[_].item()
    # 1,256,256

    pic = color[result.long()].permute(0,3,1,2)
    pic = torch.cat([pic,color[data.labels[idx][1:2]].permute(0,3,1,2)],dim=0)
    save_image(pic,fp=f'test/test1/{i}.png')'''

