from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .maskrcnn import resnet50,resnext50_32x4d,ResNet50_Weights,ResNeXt50_32X4D_Weights
from .net_utils import DenseBlock,remove_small_objects,get_bounding_box
from skimage.segmentation import watershed
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
import numpy as np
import cv2


class HoverNet(nn.Module):
    def __init__(self,num_class=7,backbone=resnet50(ResNet50_Weights.IMAGENET1K_V1)) -> None:
        super().__init__()
        self.num_class = num_class
        self.backbone = backbone
        def forward(self,x:torch.Tensor):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
            return x1, x2, x3, x4
        self.backbone.forward = forward.__get__(self.backbone, type(self.backbone))
        del self.backbone.fc,self.backbone.avgpool
        self.backbone.conv1 = nn.Conv2d(3,64,7,1,3)
        self.conv_bot = nn.Conv2d(2048, 1024, 1, stride=1, padding=0, bias=False)

        ksize=5
        def create_decoder_branch(out_ch=2,ksize=5):
            pad = ksize//2
            module_list = [ 
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),),
            ]

            u3 = nn.Sequential(OrderedDict(module_list))
            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),),
            ]
            u1 = nn.Sequential(OrderedDict(module_list))

            module_list = [ 
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),),
            ]
            u0 = nn.Sequential(OrderedDict(module_list))

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0),])
            )
            return decoder
        
        self.decoder = nn.ModuleDict(
            OrderedDict(
                [
                        ("tp", create_decoder_branch(ksize=ksize, out_ch=num_class)),
                        ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
                ]
            )
        )
        self.up = nn.Upsample(scale_factor=2)
        #self.weight_init()


    def weight_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            # ! Fixed the type checking
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

            if "norm" in classname.lower():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            if "linear" in classname.lower():
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, imgs):
        d0, d1, d2, d3 = self.backbone(imgs)
        d3 = self.conv_bot(d3)
        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.up(d3) + d2
            u3 = branch_desc[0](u3)

            u2 = self.up(u3) + d1
            u2 = branch_desc[1](u2)

            u1 = self.up(u2) + d0
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict 

def proc_np_hv(pred):
    """Process Nuclei Prediction with XY Coordinate Map.

    Args:
        pred: prediction output, assuming 
              channel 0 contain probability map of nuclei
              channel 1 containing the regressed X-map
              channel 2 containing the regressed Y-map

    """
    pred = np.array(pred, dtype=np.float32)
    # h,w
    blb_raw = pred[..., 0]
    h_dir_raw = pred[..., 1]
    v_dir_raw = pred[..., 2]

    # processing
    #0,1
    blb = np.array(blb_raw >= 0.5, dtype=np.int32)
    # 联通域分割
    blb = measurements.label(blb)[0]
    blb = remove_small_objects(blb, min_size=10)
    # h,w
    blb[blb > 0] = 1  # background is 0 already

    h_dir = cv2.normalize(
        h_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )
    v_dir = cv2.normalize(
        v_dir_raw, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
    )

    # 求梯度
    sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=21)
    sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=21)

    sobelh = 1 - (
        cv2.normalize(
            sobelh, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )
    sobelv = 1 - (
        cv2.normalize(
            sobelv, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
    )

    overall = np.maximum(sobelh, sobelv)
    overall = overall - (1 - blb)# 对前景考虑梯度
    overall[overall < 0] = 0 

    #距离图，中心的值大
    dist = (1.0 - overall) * blb
    ## nuclei values form mountains so inverse to get basins
    dist = -cv2.GaussianBlur(dist, (3, 3), 0)

    # 阈值，大于0.4的作为正例选出细胞中心
    overall = np.array(overall >= 0.4, dtype=np.int32)
    # 
    marker = blb - overall
    marker[marker < 0] = 0
    marker = binary_fill_holes(marker).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
    marker = measurements.label(marker)[0]
    marker = remove_small_objects(marker, min_size=10)

    proced_pred = watershed(dist, markers=marker, mask=blb)

    return proced_pred
