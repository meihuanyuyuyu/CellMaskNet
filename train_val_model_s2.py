from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, lr_scheduler
import torch
from torchvision.utils import save_image,draw_bounding_boxes,make_grid
from model.maskrcnn import MaskRCNN
from config import arg,anchors_wh,color
from tools.utils import generate_anchors,rois2img
from tools.augmentation import My_colorjitter,Random_flip
from tools.dataset import ConicDataset,DataLoader,Subset,collect_fn
from tools.metric import proposal_stage2_metric
from tqdm import tqdm
import os
import gc

# To get stage1 Boxes:
def stage1_boxes_hook_for_val(moudle,input,output):
    global stage1_rois
    stage1_rois = input[1]
stage1_rois = None

torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True

print('Check setted Config:\n')
for _ in dir(arg):
    if not _.startswith('_'):
        print(_,'=',getattr(arg,_))

write= SummaryWriter(arg.tensorboard_dir)

data = ConicDataset(transfs=[Random_flip(),My_colorjitter()])
indexes = torch.load('train_test_indexes/splitted_indexes.pt')
train_indexes = indexes['train']
test_indexes=indexes['test']
train_set = Subset(data,train_indexes)
test_set = Subset(data,test_indexes)
train_loader = DataLoader(train_set,2,True,num_workers=4,collate_fn=collect_fn)
test_loader = DataLoader(test_set,2,shuffle=False,num_workers=4,collate_fn=collect_fn)

######## anchors ###############
anchors = generate_anchors(anchors_wh,256,4).cuda()
net = arg.model(anchors,stage1_mode=arg.stage1_mode,stage2_train_mode=arg.stage2_train_mode,rpn_pos_threshold=arg.rpn_pos_thresh,stage2_sample_ratio=arg.stage2_sample_ratio,post_rpn_thresh=arg.post_rpn_pos_thresh,post_detection_iou_thresh=arg.post_detection_nms_thresh,post_detection_score_thresh=arg.post_detection_score_thresh).cuda()
net.load_state_dict(torch.load(arg.load_model_para))
optimizer = AdamW(net.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
lr_s = lr_scheduler.MultiStepLR(optimizer,arg.multi_steps,0.5)
net.box_align.register_forward_hook(stage1_boxes_hook_for_val)

########### fixing the middle layer gradient ####################
if not arg.stage1_mode:
    for parameters in net.backbone.parameters():
        parameters.requires_grad = False
    net.rpn.requires_grad_(False)

for _ in range(arg.epoch):
    bar = tqdm(train_loader,colour='CYAN')
    net.train()
    losses = []
    for data in bar:
        imgs,labels,boxes,masks,cls = data
        imgs = imgs.cuda()
        masks = [mask.cuda() for mask in masks]
        boxes = [box.cuda() for box in boxes]
        cls = [_.cuda() for _ in cls]
        loss = net(imgs,boxes,masks,cls)
        optimizer.zero_grad()

        if  loss['detection_loss'] is not None:
            federal_loss=loss['detection_loss']+loss['rpn_loss']
        else:
            # stage1 predicts nothing
            federal_loss=loss['rpn_loss']
        federal_loss.backward()
        optimizer.step()
        lr_s.step()
        if not federal_loss.isnan().any():
            losses.append(federal_loss)
        bar.set_description(f"stage2 train loss:{federal_loss.item()}")
    write.add_scalar('train loss', torch.tensor(losses).mean().item(), _)
    torch.save(net.state_dict(),arg.model_para)
    torch.cuda.empty_cache()
    
    if _% 10 ==0:
        with torch.no_grad():
            gc.collect()
            net.eval()
            boxes_ious = []
            masks_ious = []
            cls_acc = []
            bar = tqdm(test_loader,colour='red')
            for i,data in enumerate(bar):
                imgs,labels,boxes,masks,cls = data
                imgs = imgs.cuda()
                labels = labels.cuda()
                masks = [mask.cuda() for mask in masks]
                boxes = [box.cuda() for box in boxes]
                cls = [_.cuda() for _ in cls]
                rois,score,out_cls,out_masks=net(imgs)
                ######## mask iou/boxes iou/category accurcay###################################################################
                boxes_iou,masks_jaccard,categ_accuracy=proposal_stage2_metric(rois,out_masks,out_cls,boxes,masks,cls)
                boxes_ious.append(boxes_iou)
                masks_ious.append(masks_jaccard)
                cls_acc.append(categ_accuracy)
                bar.set_description(f'masks iou:{masks_jaccard},boxes iou:{boxes_iou},category acc:{categ_accuracy}')
                ########### visuliazation: label with bounding boxes and rois2img with bouding boxes ############################
                # stage2 predicted picture
                pic = rois2img(rois,out_cls,out_masks)
                pic = color[pic].permute(0,3,1,2)  
                target = color[labels[:,1]].permute(0,3,1,2)
                target1 = torch.zeros_like(target)
                target2 = torch.zeros_like(target)
                # stage1 boxes predicted picture:
                for _,(per_img,stage1_boxes,boxes) in enumerate(zip(target,stage1_rois,rois)):
                    target1[_]=draw_bounding_boxes((per_img*255).to(dtype=torch.uint8),stage1_boxes)
                    target2[_] = draw_bounding_boxes((per_img*255).to(dtype=torch.uint8),boxes)
                pic =make_grid(torch.cat([(target1/255).float(),(target2/255).float(),pic]),nrow=2,padding=4)
                save_image(pic,os.path.join(arg.val_img_fp,f'{i}.png'))
            write.add_scalar('boxes miou', torch.tensor(boxes_ious).mean().item(), _)
            write.add_scalar('masks miou', torch.tensor(masks_ious).mean().item(), _)
            write.add_scalar('category accuracy', torch.tensor(cls_acc).mean().item(), _)