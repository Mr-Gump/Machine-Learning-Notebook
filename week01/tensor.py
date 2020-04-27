import torch,time
import numpy as np

'''
exercise 1
flag = True

if flag:
    arr = np.ones((3,3))
    print('ndarry的类型是:' ,arr.dtype)

t1 = time.time()
t = torch.tensor(arr,device='cuda')
t2 = time.time()
print('使用gpu消耗时间为:',str(t2-t1))
print(t)

t1 = time.time()
t = torch.tensor(arr)
t2 = time.time()
print('使用cpu消耗的时间为:',str(t2-t1))
print(t)'''

'''
exercise 2
flag = True

if flag:
    arr = np.array([[1,2,3],[4,5,6]])
    t = torch.from_numpy(arr)
    print('ndarry:',arr)
    print('tensor:',t)

    # 修改arr第一个数据为0
    arr[0,0] = 0
    print('ndarry:',arr)
    print('tensor:',t)

    # 修改tensor第一个数据改为-1
    t[0,0] = -1
    print('ndarry:',arr)
    print('tensor:',t)'''

'''
exercise 3
flag = True

if flag:
    out_t = torch.tensor([1])

    t = torch.zeros((3,3) , out = out_t)
    print(t,'\n',out_t)
    print(id(t),id(out_t),id(t)==id(out_t))'''

flag = True

if flag:
    # full
    print(torch.full((3,3),10))
    # arrage
    print(torch.arange(0,10,2))
    # linspace
    print(torch.linspace(0,10,6))
    # logspace
    print(torch.logspace(10,100,5,10))
    # eye
    print(torch.eye(5))