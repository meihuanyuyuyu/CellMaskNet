from .maskrcnn import MaskRCNN, forward
import torch.nn as nn
import torch
from typing import List
from torchvision.ops import roi_align

class AdaptiveFeaturePooling(nn.Module):
    def __init__(self,boxes:List[torch.tensor],mode:str='max') -> None:
        super().__init__()
        self.boxes = boxes
        self.mode = mode

    def forward(self,feature_maps:List[torch.tensor]):
        for feature_map in feature_maps:
            roi_align()