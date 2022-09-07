from cProfile import label
from torchvision.ops import masks_to_boxes,remove_small_boxes,roi_align
from torch.nn.functional import one_hot,grid_sample
import torch
import numpy as np
from torch.utils.data import Dataset,Subset,DataLoader
from tools.utils import box2grid, remove_big_boxes
from torchvision.transforms.functional import crop



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