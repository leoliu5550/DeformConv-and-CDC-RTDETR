import torch
import torch.nn as nn
import copy

class net(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10,10)

        self.layers = nn.ModuleList(
            [copy.deepcopy(self.linear) for _ in range(3)]
        )

    def forward(self,x):
        out = []
        for x_,l_ in zip(x,self.layers):
            out.append(l_(x_))
            pass
            
        return out
    
x= [torch.ones(1,10) for _ in range(3)]
print(x)
net2 =net()
print(net2(x))