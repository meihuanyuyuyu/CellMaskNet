cd /home/DM19/workspace/Cell_Mask_RCNN
python3 hovernet_train_val.py --stage_mode=2 >> log/log7.txt 2>&1

#python3 inference.py 
#python3 project_conic/rules/compute_stats.py --mode=seg_class --pred=numpy_prediction/Config5.npy --true=numpy_prediction/Config5_gt.npy
curl http://172.17.0.1:9999/gpumanager/stopearlycontainer?user=DM19
