import torch
from model.maskrcnn import MaskRCNN

anchors_wh = torch.tensor([[6.3531, 4.2322], [5.0907, 7.3089], [7.9669, 7.3888], [11.1108, 7.9923], [8.0002, 11.2820], [12.6215, 12.2743], [15.7811, 8.1808], [8.5952, 15.9470], [17.4354, 13.5288],
                               [13.9092, 17.6629], [22.8538, 10.4661], [12.1536, 24.4449], [21.0220, 19.6782], [32.4152, 15.9982], [22.7655, 33.5465]],
                              device='cpu')
color = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]], dtype=torch.float32, device='cuda')

class Config1():
    tensorboard_dir = 'exp_data/MaskRCNN'
    model_para = 'model_parameters/maskrcnn.pt'
    lr = 2e-3
    weight_decay = 1e-5
    multi_steps: list = [30, 70, 100]
    epoch = 120
    rpn_pos_thresh = 0.7

    stage1_mode = True
    stage2_train_mode = False

class Config2(Config1):
    model = MaskRCNN

    tensorboard_dir = 'exp_data/MaskRCNNStage2'
    model_para = 'model_parameters/maskrcnn_stage2.pt'
    val_img_fp = 'figures/maskrcnn_s2'

    rpn_pos_thresh = 0.7
    post_rpn_pos_thresh = 0.7
    post_detection_score_thresh = 0.5
    post_detection_nms_thresh = 0.2
    stage2_sample_ratio = 0.5

    stage1_mode = False
    stage2_train_mode = True

class Config3(Config2): 
    'pq       multi_pq+   0.443299    0.059354 '
    tensorboard_dir = 'exp_data/MaskRCNNStage3'
    model_para = 'model_parameters/maskrcnn_stage3.pt'
    load_model_para = 'model_parameters/maskrcnn_stage2.pt'
    val_img_fp = 'figures/maskrcnn_s3'
    numpy_dir = 'numpy_prediction'
    rpn_pos_thresh= 0.6
    post_rpn_pos_thresh = 0.7
    stage1_mode = True
    stage2_train_mode = True
    stage2_sample_ratio = 3

class Config4(Config3):
    'config3训练分类坍缩,try:调整二阶段采样率，损失为分类增加权重'
    stage2_sample_ratio = 2
    loss_cls_weight = 1.5
    lr = 1e-3
    model_para = 'model_parameters/maskrcnn_stage3_config4.pt'
    load_model_para = 'model_parameters/maskrcnn_stage3.pt'
    val_img_fp = 'figures/maskrcnn_s3_config4'
    tensorboard_dir = 'exp_data/MaskRCNNStage3_config4'

class Config5(Config4):
    '固定住COnfig3中一阶段权重，进行单独二阶段训练'
    stage1_mode = False
    stage2_train_mode = True
    model_para = 'model_parameters/maskrcnn_stage3_config5.pt'
    load_model_para = 'model_parameters/maskrcnn_stage3.pt'
    val_img_fp = 'figures/maskrcnn_s3_config5'
    tensorboard_dir = 'exp_data/MaskRCNNStage3_config5'                                    

arg = Config5()
