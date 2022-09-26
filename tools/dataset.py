from cProfile import label
from torchvision.ops import masks_to_boxes,remove_small_boxes,roi_align
from torch.nn.functional import one_hot,grid_sample
import torch
import numpy as np
from torch.utils.data import Dataset,Subset,DataLoader
from tools.utils import box2grid, remove_big_boxes
from torchvision.transforms.functional import crop
from scipy.ndimage import center_of_mass



class ConicDataset(Dataset):
    def __init__(self,masks_size=28,transfs:list=None) -> None:
        super().__init__()
        self.transfs = transfs
        self.masks_size = masks_size
        print('Loading data into memory')
        self.imgs = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous()
        self.labels = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float64)).long().permute(0,3,1,2).contiguous()
        print('Finishing loading data')

    def transforms(self,imgs,labels):
        for _ in self.transfs:
            imgs,labels = _(imgs,labels)
            return imgs,labels
    
    def generate_targets_boxes(self,labels: torch.Tensor, mask_size: int = 28):
        max_instance = labels[0].max()
        if max_instance== 0:
            return torch.zeros(0,4),torch.zeros(0,mask_size,mask_size),torch.zeros(0)
        # -----------------------boxes------------------------------
        instances = one_hot(labels[0], max_instance + 1).permute(2, 0, 1)[1:] #n,h,w
        boxes = masks_to_boxes(instances)
        # enlarge box boundary
        boxes[:, 2:] = boxes[:, 2:] + 1
        boxes[:, :2] = boxes[:, :2] - 1
        index = remove_small_boxes(boxes, 1.5)
        instances = instances[index]
        boxes = boxes[index]
        index = remove_big_boxes(boxes, 70)
        boxes = boxes[index]
        instances = instances[index]  # num,256,256
        #------------------------masks--------------------------------
        #grids = box2grid(boxes,[mask_size,mask_size])
        index = torch.arange(0,len(boxes),device=boxes.device).unsqueeze(1)
        rois = torch.cat([index,boxes],dim=1)
        masks=roi_align(labels[None,1:]*instances.unsqueeze(1).float(),rois,output_size=mask_size,spatial_scale=1,sampling_ratio=1,aligned=True).round().long().squeeze(1)
        #masks = grid_sample(labels[None,1:].float()*instances.unsqueeze(1),grids,align_corners=True).round().long().squeeze(1) #rois,28,28
        #masks =crop(labels[None,1:].float()*instances.unsqueeze(0),boxes[:])
        cls = one_hot(masks,num_classes=7).permute(0,3,1,2).sum(dim=[2,3])[:,1:].argmax(-1)+1
        return boxes, masks.bool().long(), cls
    
    def __getitem__(self, index):
        img = self.imgs[index]
        label = self.labels[index]
        if self.transfs is not None:
            img,label = self.transforms(img,label)
        boxes,masks,cls=self.generate_targets_boxes(label,mask_size=self.masks_size)
        return img,label,boxes,masks,cls
    
def collect_fn(data):
    return [torch.stack(sub_data,dim=0) if _== 0 or _==1 else list(sub_data) for _,sub_data in enumerate(zip(*data))] 


class HoverNetCoNIC(Dataset):
    def __init__(self,img_size:list,transf:list) -> None:
        super().__init__()
        self.grid_x,self.grid_y = torch.meshgrid(torch.arange(img_size[0]), torch.arange(img_size[1]), indexing='xy')
        self.transf = transf
        print('Loading data into memory')
        self.imgs = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous()
        self.labels = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/labels.npy').astype(np.float64)).long().permute(0,3,1,2).contiguous()
        print('Finishing loading data')
        

    def transforms(self,imgs,labels):
        for _ in self.transf:
            imgs,labels = _(imgs,labels)
            return imgs,labels
    
    def hv_label_generator(self,label:torch.Tensor):
        'label:2,h,w'
        hv = torch.zeros_like(label,dtype=torch.float32)
        instance_map = label[0]
        for idx in range(1,instance_map.max()+1):
            mask = instance_map==idx
            mask_np = mask.numpy()
            center_y,center_x = center_of_mass(mask_np)
            #center_x,center_y = torch.from_numpy(center_x),torch.from_numpy(center_y)
            x = self.grid_x[mask]-center_x
            x[x<0] = x[x<0]/ x.min().abs()
            x[x>0] = x[x>0]/ x.max()
            hv[0,mask] = x

            y = self.grid_y[mask]- center_y
            y[y<0] = y[y<0]/ y.min().abs()
            y[y>0] = y[y>0]/ y.max()
            hv[1,mask] = y
        return hv
    
    def __getitem__(self, index):
        label = self.labels[index]
        img = self.imgs[index]
        if len(self.transf) !=0:
            img,label=self.transforms(img,label)
        hv =self.hv_label_generator(label)
        np = label[0].bool().long()
        nc = label[1]
        return img,hv,np,nc
