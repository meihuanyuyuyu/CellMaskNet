from operator import index
import torch
import numpy as np
from model.maskrcnn import MaskRCNN
#from Config.maskrcnn_config import arg,anchors_wh
from tools.utils import draw_instance_map,convert_prediction_to_numpy
import os                 
from project_conic.rules.compute_pq_mpq import compute_pq_mpq
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='infering')
parser.add_argument('--model',type=str,default='cellmasknet')
parser.add_argument('--device',type=str,default='cuda')
parser.add_argument('--dataset',type=str,default='conic')
parser.add_argument('--results_dir',type=str,default='figures/results/hovernet')
process_arg = parser.parse_args()

'''@torch.no_grad()
def get_final_metric():
    anchors = generate_anchors(anchors_wh,img_size=256,scale=4).cuda()
    net = MaskRCNN(anchors,rpn_pos_threshold=arg.rpn_pos_thresh,stage1_mode=False,stage2_train_mode=False,post_rpn_thresh=arg.post_rpn_pos_thresh,).cuda()
    net.eval()
    net.load_state_dict(torch.load(arg.model_para))

    imgs = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous().cuda()
    gts = np.load('project_conic/CoNIC_Challenge/labels.npy')
    test_index = torch.load('train_test_indexes/splitted_indexes.pt')['test']

    preds = torch.zeros(0, 2, 256, 256, dtype=torch.long)
    _gts = np.zeros((0, 256, 256, 2), dtype=np.uint16)
    for i,index in enumerate(tqdm(test_index)):
        img = imgs[index:index+1]
        gt =gts[index:index+1]
        rois,scores,cls,masks =net(img)
        rois,scores,cls,masks = rois[0],scores[0],cls[0],masks[0]
        pred =convert_prediction_to_numpy(rois,cls,masks)
        preds = torch.cat([preds,pred],dim=0)
        _gts = np.concatenate((_gts,gt),axis=0)
    preds = preds.permute(0,2,3,1).numpy().astype(np.uint16)
    np.save(os.path.join(arg.numpy_dir,arg.__class__.__name__),preds)
    np.save(os.path.join(arg.numpy_dir,arg.__class__.__name__+'_gt'),_gts)'''


class InferEngine:
    def __init__(self) -> None:
        print('initlizing model...')
        if process_arg.model == 'hovernet':
            from hovernet_inference import HovernetInfer
            self.infer = HovernetInfer(process_arg.device,process_arg.dataset)
        if process_arg.model == 'maskrcnn':
            from maskrcnn_inference import MaskRCNNInfer
            self.infer = MaskRCNNInfer(process_arg.device,process_arg.dataset)
        if process_arg.model == 'maskrcnn':
            pass
    
    def prepare_data(self):
        if process_arg.dataset == 'conic':
            data = torch.from_numpy(np.load('project_conic/CoNIC_Challenge/images.npy').astype(np.float64)/255).float().permute(0,3,1,2).contiguous()
            gts = np.load('project_conic/CoNIC_Challenge/labels.npy')
            index =  torch.load('train_test_indexes/splitted_indexes.pt')['test']
            return data,gts,index

    def generate_results(self,data:torch.Tensor,gts,index:list):
        preds = torch.zeros(0, 2, 256, 256, dtype=torch.long)
        _gts = np.zeros((0, 256, 256, 2), dtype=np.uint16)
        for _,idx in enumerate(tqdm(index)):
            img = data[idx:idx+1]
            gt = gts[idx:idx+1]
            pred =self.infer(img)
            preds = torch.cat([preds,pred],dim=0)
            _gts = np.concatenate((_gts,gt),axis=0)
        preds = preds.permute(0,2,3,1).numpy().astype(np.uint16)
        return preds,_gts
    

    def run(self):
        data,gts,index = self.prepare_data()
        preds,_gts=self.generate_results(data,gts,index)
        pred_path = os.path.join(self.infer.arg.numpy_dir,self.infer.arg.__class__.__name__)+'.npy'
        true_path = pred_path+'_gt.npy'
        draw_instance_map(data,preds,index,process_arg.results_dir)
        np.save(pred_path,preds)
        np.save(true_path,_gts)
        compute_pq_mpq(pred_path,true_path)

  


if __name__ =='__main__':
    engine = InferEngine()
    engine.run()


        




    
    
