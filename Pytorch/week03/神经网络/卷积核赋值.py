import torch
import torch.nn as nn

t1 = torch.full((5,5),1,dtype=torch.float32)
t2 = torch.full((5,5),2,dtype=torch.float32)
t3 = torch.full((5,5),3,dtype=torch.float32)

t = torch.stack([t1,t2,t3],dim=0)

t.unsqueeze_(0)

print('原始张量:',t)
print('原始大小:',t.shape)

k1 = torch.full((3,3),1,dtype=torch.float32)
k2 = torch.full((3,3),2,dtype=torch.float32)

cnv = nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,padding=0,stride=1,bias=0)

# 给两个卷积核赋值
cnv.weight.data[0,:,:,:] = torch.full((3,3),1,dtype=torch.float32)
cnv.weight.data[1,:,:,:] = torch.full((3,3),2,dtype=torch.float32)

print('输出张量为:',cnv(t))
print('输出张量大小为:',cnv(t).shape)

cnv = nn.Conv2d(in_channels=3,out_channels=2,kernel_size=3,padding=1,padding_mode='zeros',stride=1,bias=0)

cnv.weight.data[0,:,:,:] = torch.full((3,3),1,dtype=torch.float32)
cnv.weight.data[1,:,:,:] = torch.full((3,3),2,dtype=torch.float32)

print('输出张量为:',cnv(t))
print('输出张量大小为:',cnv(t).shape)