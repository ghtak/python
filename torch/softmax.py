'''
다중 클래스 분류 -> 선택지가 정답일 확률 

H(x)=softmax(Wx+b)

x = class x feature
w = feature x pred
b = class x pred
y = class x pred

y = softmax(xw+b)

class=5, fature=4, pred=3

X=5x4
w=4x3
b=5x3
Y=5x3

Y(5x3) = softmax(X(5x4) x w(4x3) + b(5x3))

cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
'''

import torch
import torch.nn.functional as F


def softmax_cost():
    z = torch.FloatTensor([1,2,3])
    h = F.softmax(z, dim=0)
    print(h, h.sum())

    #---
    z = torch.rand(3,5, requires_grad=True)
    h = F.softmax(z, dim=1)
    print(h, h.sum())

    y = torch.randint(5,(3,)).long()
    print(y,y.size(), y.unsqueeze(1))

    y_one_hot = torch.zeros_like(h)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    print(y_one_hot)


if __name__ == '__main__':
    torch.manual_seed(1)
    softmax_cost()
 