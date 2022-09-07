from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW, lr_scheduler
from torch.nn.functional import cross_entropy, smooth_l1_loss, binary_cross_entropy
import torch
from torchvision.utils import save_image
from model.maskrcnn import MaskRCNN
from config import arg,anchors_wh
from tools.utils import generate_anchors
from tools.augmentation import My_colorjitter,Random_flip
from tools.dataset import ConicDataset,DataLoader,Subset,collect_fn
from tools.metric import stage1_val
from tqdm import tqdm



torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.benchmark = True


def stage1_boxes_hook_for_val(moudle,input,output):
    global stage1_rois
    stage1_rois = input[1]


write= SummaryWriter(arg.tensorboard_dir)

data = ConicDataset(transfs=[Random_flip(),My_colorjitter()])
indexes = torch.load('train_test_indexes/splitted_indexes.pt')
test_indexes=indexes['test']
train_indexes = indexes['train']
train_set = Subset(data,train_indexes)
test_set = Subset(data,test_indexes)
train_loader = DataLoader(train_set,2,True,num_workers=4,collate_fn=collect_fn)
test_loader = DataLoader(test_set,2,shuffle=False,num_workers=4,collate_fn=collect_fn)

######## anchors ###############
anchors = generate_anchors(anchors_wh,256,4).cuda()
net = MaskRCNN(anchors,stage1_mode=True,rpn_pos_threshold=0.7).cuda()
optimizer = AdamW(net.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
lr_s = lr_scheduler.MultiStepLR(optimizer,arg.multi_steps,0.5)

# using hook to get stage1 proposals
stage1_rois = None
net.box_align.register_forward_hook(stage1_boxes_hook_for_val)

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
        loss['rpn_loss'].backward()
        optimizer.step()
        lr_s.step()
        if not loss['rpn_loss'].isnan().any():
            losses.append(loss['rpn_loss'])
        bar.set_description(f"stage1 train {loss['rpn_loss'].item()}")
    write.add_scalar('train loss', torch.tensor(losses).mean().item(), _)

    torch.save(net.state_dict(),arg.model_para)
    
    if _% 10 ==4:
        with torch.no_grad():
            net.eval()
            ious = []
            bar = tqdm(test_loader,colour='red')
            for i,data in enumerate(bar):
                imgs,labels,boxes,masks,cls = data
                imgs = imgs.cuda()
                labels = labels.cuda()
                boxes = [box.cuda() for box in boxes]
                rois,score,out_cls,out_reg,out_masks=net(imgs)
                iou = stage1_val(boxes,stage1_rois,imgs,f'figures/maskrcnn_s1/{i}.png')
                if iou is not None:
                    bar.set_description(f'boxes iou:{iou}')
                    ious.append(iou)
            write.add_scalar('Test iou', torch.tensor(ious).mean().item(), _)
                
