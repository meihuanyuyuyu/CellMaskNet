
from torch.utils import tensorboard
from Config.hovernet_config import arg
import argparse
import torch
import logging

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.functional import cross_entropy,mse_loss
from tools.augmentation import Random_flip,Randomrotation,ColorJitter,MyGausssianBlur
from tools.dataset import HoverNetCoNIC,DataLoader,Subset
from tqdm import tqdm

parser = argparse.ArgumentParser(description='train hovernet')
parser.add_argument('--stage_mode',type=int,default=1)
parser.add_argument('--device',type=str,default='cuda')
process_arg = parser.parse_args()


def train_hovernet(data:DataLoader,model:torch.nn.Module):
    model.train()
    bar = tqdm()
    

def val_hovernet():
    pass

def process():
    pass

if __name__=='__main__':
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    
    logging.basicConfig(filename=arg.log_dir,level=logging.INFO,format='|%(asctime)s.%(msecs)03d| [%(levelname)s] %(message)s',datefmt='%Y-%m-%d|%H:%M:%S')
    logging.info('loading model ...')
    net = arg.model(arg.num_classes)
    optimizer = Adam(net.parameters(),lr=arg.lr,weight_decay=arg.weight_decay)
    lrs = MultiStepLR(optimizer,[25],gamma=0.1)

    logging.info('prepare dataset')
    data_set = HoverNetCoNIC(img_size=arg.img_size,transf=[Random_flip(),MyGausssianBlur(3,(0.2,0.5)),ColorJitter(0.1,0.1,0.1,0.1)])
    indexes = torch.load('train_test_indexes/splitted_indexes.pt')
    train_indexes,test_indexes = indexes['train'],indexes['test']
    train_set=Subset(data_set,train_indexes)
    test_set = Subset(data_set,test_indexes)
    train_data = DataLoader(train_set,batch_size=arg.batch,shuffle=True,num_workers=4)
    test_data = DataLoader(test_set,batch_size=arg.batch,shuffle=False,num_workers=4)

