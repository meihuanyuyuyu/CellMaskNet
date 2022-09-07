cd /home/DM19/workspace/Cell_Mask_RCNN
python3 train_val_model_s2.py >> log.txt 2>&1
curl http://172.17.0.1:9999/gpumanager/stopearlycontainer?user=DM19
