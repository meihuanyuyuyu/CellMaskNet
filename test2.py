import torch
a = torch.rand(1,4,2).float()
b = torch.rand(1,4,2).float()
print(torch.max(torch.cat([a,b]),dim=0).values)
