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
import torch.nn as nn
import torch.optim as optim


def softmax_cost():
    z = torch.FloatTensor([1, 2, 3])
    h = F.softmax(z, dim=0)
    print(h, h.sum())

    # ---
    z = torch.rand(3, 5, requires_grad=True)
    print(z)
    h = F.softmax(z, dim=1)
    print(h, h.sum())

    y = torch.randint(5, (3,)).long()
    print(y, y.size(), y.unsqueeze(1))

    y_one_hot = torch.zeros_like(h)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    print(y_one_hot)

    # cost = (y_one_hot * -torch.log(h)).sum(dim=1).mean()
    # =>
    # cost = (y_one_hot * -F.log_softmax(z, dim=1)).sum(dim=1).mean()
    # =>
    # cost = F.nll_loss(F.log_softmax(z, dim=1), y)
    # =>
    cost = F.cross_entropy(z, y)
    print(cost)


def softmax_train():
    x_train = [[1, 2, 1, 1],
               [2, 1, 3, 2],
               [3, 1, 3, 4],
               [4, 1, 5, 5],
               [1, 7, 5, 5],
               [1, 2, 5, 6],
               [1, 6, 6, 6],
               [1, 7, 7, 7]]
    y_train = [2, 2, 2, 1, 1, 1, 0, 0]
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    # x = 8x4
    # class = 3
    # y_one_hot = 8x3
    y_one_hot = torch.zeros((8, 3))
    y_one_hot.scatter_(1, y_train.unsqueeze(1), 1)
    print(x_train, y_one_hot)

    W = torch.zeros((4, 3), requires_grad=True)
    b = torch.zeros(3, requires_grad=True)
    print(b)
    optimizer = optim.SGD([W, b], lr=0.1)
    nb_epochs = 1000
    for epoch in range(nb_epochs+1):
        out = F.softmax(x_train.matmul(W) + b, dim=1)
        cost = (y_one_hot * -torch.log(out)).sum(dim=1).mean()
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))



def softmax_train1():
    x_train = [[1, 2, 1, 1],
               [2, 1, 3, 2],
               [3, 1, 3, 4],
               [4, 1, 5, 5],
               [1, 7, 5, 5],
               [1, 2, 5, 6],
               [1, 6, 6, 6],
               [1, 7, 7, 7]]
    y_train = [2, 2, 2, 1, 1, 1, 0, 0]
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)
    
    '''
    1 raw
    W = torch.zeros((4, 3), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = optim.SGD([W, b], lr=0.1)
    '''

    model = nn.Linear(4,3)
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    print(list(model.parameters()))

    nb_epochs = 1000
    for epoch in range(nb_epochs+1):
        # 1 raw pred = x_train.matmul(W) + b
        pred = model(x_train)
        cost = F.cross_entropy(pred, y_train)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))


if __name__ == '__main__':
    torch.manual_seed(1)
    softmax_train1()
