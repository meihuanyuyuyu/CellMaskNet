from model.HoverNet import HoverNet
import os

class Config:
    img_size = [256,256]
    num_classes = 7
    lr = 1e-4
    lr_s = [25]
    weight_decay = 1e-4
    batch = 8
    model = HoverNet
    epoch = 50
    
    def __init__(self) -> None:
        self.log_dir =f'log/{self.__class__.__name__}.log'
        self.exp_data_dir = f'exp_data/{self.model.__name__}'
        if not os.path.exists(self.exp_data_dir):
            os.makedirs(self.exp_data_dir)
        self.val_img_fp = f'figures/{self.model.__name__}_{self.__class__.__name__}'
        if not os.path.exists(self.val_img_fp):
            os.makedirs(self.val_img_fp)
        self.model_para = f'model_parameters/{self.model.__name__}_{self.__class__.__name__}.pt'

arg = Config()