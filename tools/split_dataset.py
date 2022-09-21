import torch

def split_dataset(data:torch.Tensor,fp,train_set:float=0.7,test_set:float=0.2,val_set:float=0.1):
    length = len(data)
    train,test,val=torch.randperm(length).split([int(train_set*length),int(test_set*length),length-int(train_set*length)-int(test_set*length)])
    index = {}
    index.update({'train':train,'test':test,'val':val})
    torch.save(index,fp)
