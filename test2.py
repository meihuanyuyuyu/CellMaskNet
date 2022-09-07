<<<<<<< HEAD
from config import arg

print(dir(arg))
for _ in dir(arg):
    if not _.startswith('_'):
        print(_,'=',getattr(arg,__package__))
=======
import torch
a = torch.rand(1,4,2).float()
b = torch.rand(1,4,2).float()
print(torch.max(torch.cat([a,b]),dim=0).values)
>>>>>>> 9696916690c69616ae0f1825a8817e27a632a22e
