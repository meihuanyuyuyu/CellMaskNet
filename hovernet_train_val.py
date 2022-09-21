

from torch.utils.tensorboard import SummaryWriter
from Config.hovernet_config import arg
import argparse
import torch
import logging

from torch.optim import Adam,Optimizer
from torch.optim.lr_scheduler import MultiStepLR,_LRScheduler
from torch.nn.functional import cross_entropy,mse_loss
from tools.augmentation import Random_flip,Randomrotation,ColorJitter,MyGausssianBlur
from tools.dataset import HoverNetCoNIC,DataLoader,Subset
from tools.metric import jaccard,mask_category_acc
from tqdm import tqdm
import matplotlib.pyplot as plt 

parser = argparse.ArgumentParser(description='train hovernet')
parser.add_argument('--stage_mode',type=int,default=1)
parser.add_argument('--device',type=str,default='cuda')
process_arg = parser.parse_args()

def train_hovernet(data:DataLoader,optimizer:Optimizer,model:torch.nn.Module,lr_s:_LRScheduler):
    model.train()
    bar = tqdm(data,colour='CYAN')
    losses = []
    for data in bar:
        img,hv,np,tp = data
        img = img.to(device=process_arg.device)
        hv = hv.to(device=process_arg.device)
        np = np.to(device=process_arg.device)
        tp =  tp.to(device=process_arg.device)
        out_dict =model(img)
        loss = cross_entropy(out_dict['tp'],tp) + cross_entropy(out_dict['np'],np) + mse_loss(out_dict['hv'],hv)
        iou = jaccard(out_dict['np'].argmax(1),np)
        acc = mask_category_acc(out_dict['tp'].argmax(1),tp)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_s.step()
        losses.append(loss)
        bar.set_description(f'loss:{loss.item():.4f},masks iou:{iou:.4f},category acc:{acc:.4f}')
    return torch.tensor(losses).mean().item()

    

def val_hovernet(data:DataLoader,optimizer:Optimizer,model:torch.nn.Module):
    model.eval()
    bar = tqdm(data,colour='red')
    for data in bar:
        img,hv,np,tp = data
        hv = hv.to(device=process_arg.device)
        np = np.to(device=process_arg.device)
        tp =  tp.to(device=process_arg.device)
        out_dict =model(img)
        plt.imshow()


    

class Training_process:
    def __init__(self) -> None:
        self.net = arg.model(arg.num_classes).to(device=process_arg.device)
        self.optimizer = Adam(self.net.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
        self.lrs = MultiStepLR(self.optimizer,[25],gamma=0.1)
        if process_arg.stage_mode ==1:
            self.net.backbone.requires_grad_(False)
        else:
            self.net.load_state_dict(torch.load(arg.model_para))
        

    def generating_data(self):
        data_set = HoverNetCoNIC(img_size=arg.img_size,transf=[Random_flip(),MyGausssianBlur(3,(0.2,0.5)),ColorJitter(0.1,0.1,0.1,0.1)])
        indexes = torch.load('train_test_indexes/splitted_indexes.pt')
        train_indexes,test_indexes = indexes['train'],indexes['test']
        train_set=Subset(data_set,train_indexes)
        test_set = Subset(data_set,test_indexes)
        train_data = DataLoader(train_set,batch_size=arg.batch,shuffle=True,num_workers=4)
        test_data = DataLoader(test_set,batch_size=arg.batch,shuffle=False,num_workers=4)
        setattr(self,'train_data',train_data)
        setattr(self,'test_data',test_data)
    
    def run(self):
        for _ in range(arg.epoch):
            losses=train_hovernet(self.train_data,optimizer=self.optimizer,model=self.net,lr_s=self.lrs)
            writer.add_scalar('train loss', losses, _)
            torch.save(self.net.state_dict(),arg.model_para)
            torch.cuda.empty_cache()


if __name__=='__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    logging.basicConfig(filename=arg.log_dir,level=logging.INFO,format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S')
    logging.info('loading model ...')
    process = Training_process()
    logging.info('prepare dataset')
    process.generating_data()
    logging.info('Tensorboard activating..')
    writer =  SummaryWriter(arg.exp_data_dir)
    process.run()


