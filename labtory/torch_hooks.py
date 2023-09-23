import torch 

a = torch.tensor(2.0,requires_grad=True)
b = torch.tensor(3.0,requires_grad=True)
c = a*b
d = torch.tensor(4.0,requires_grad=True)
e = c*d
print("#"*80)
a.register_hook(lambda grad:print(grad))
b.register_hook(lambda grad:print(grad))
c.register_hook(lambda grad:print(grad))
c.retain_grad()
e.register_hook(lambda grad:print(grad))
e.backward()
print("#"*80)
a.register_hook(lambda grad:print(grad))
b.register_hook(lambda grad:print(grad))
c.register_hook(lambda grad:print(grad))
e.register_hook(lambda grad:print(grad))