import torch
from model.HoverNet import HoverNet,proc_np_hv
from Config.hovernet_config import arg

class HovernetInfer:
    def __init__(self,device,dataset) -> None:
        self.net = HoverNet(arg.num_classes)
        self.net.load_state_dict(torch.load(arg.model_para))
        self.net.eval()
        self.net.to(device=device)
        self.dataset = dataset

    def __call__(self,img:torch.Tensor):
        if self.dataset == 'conic':
            predict:dict =self.net(img)
            tp,_np,hv = predict.values()
            hv = hv.unsqueeze(0)
            _np = _np.softmax(dim=1).max(dim=1).values.unsqueeze(0)
            tp = tp.argmax(dim=1).unsqueeze(0)
            pred = torch.stack([_np,*hv.unbind(dim=1)],dim=-1).cpu()
            proc_pred =torch.from_numpy(proc_np_hv(pred))
            pred = torch.zeros(2, *arg.img_size).long()
            pred[0] = proc_pred
            for idx in range(1, proc_pred.max() + 1):
                mask = proc_pred == idx
                cls = tp[mask].mode().values
                if cls == 0:
                    pred[0][mask] = 0
                else:
                    pred[1][mask] = cls
            return pred.unsqueeze(0)
    

        
