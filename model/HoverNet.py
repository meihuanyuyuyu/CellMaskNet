from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .maskrcnn import resnet50,resnext50_32x4d,ResNet50_Weights,ResNeXt50_32X4D_Weights
from .net_utils import DenseBlock


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
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=pad, bias=False),),
            ]

            u3 = nn.Sequential(OrderedDict(module_list))
            module_list = [ 
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=pad, bias=False),),
            ]
            u2 = nn.Sequential(OrderedDict(module_list))

            module_list = [
                ("conva/pad", nn.Conv2d(ksize=ksize, stride=1,padding=ksize//2)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False),),
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

def post_process():
    pass