'''

Hypothesis

H(x) = Wx + b

'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def test_no_zero_grad():
    w = torch.tensor(2.0, requires_grad=True)

    nb_epochs = 20
    for _ in range(nb_epochs + 1):
        z = 2*w
        z.backward()
        # 미분값은 누적 2->4->6->8...
        # optimizer.zero_grad() 로 초기화
        print('수식을 w로 미분한 값 : {}'.format(w.grad))


def linear_regression_train():
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])

    W = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([W, b], lr=0.01)

    for i in range(1000):
        hypothesis = x_train * W + b
        cost = torch.mean((hypothesis-y_train)**2)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                i, 1000, W.item(), b.item(), cost.item()
            ))


def multiple_linear_regression_train():
    x1_train = torch.FloatTensor([[73], [93], [89], [96], [73]])
    x2_train = torch.FloatTensor([[80], [88], [91], [98], [66]])
    x3_train = torch.FloatTensor([[75], [93], [90], [100], [70]])

    y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

    w1 = torch.zeros(1, requires_grad=True)
    w2 = torch.zeros(1, requires_grad=True)
    w3 = torch.zeros(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    optimizer = optim.SGD([w1, w2, w3, b], lr=1e-5)

    for i in range(1000):
        hypothesis = x1_train * w1 + x2_train * w2 + x3_train * w3 + b
        cost = torch.mean((hypothesis-y_train)**2)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch {:4d}/{} w1: {:.3f}, w2: {:.3f}, w3: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
                i, 1000,
                w1.item(),
                w3.item(),
                w1.item(),
                b.item(),
                cost.item()
            ))

    x = torch.stack([x1_train, x2_train, x3_train], dim=1).squeeze()
    w = torch.zeros((3, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = optim.SGD([w, b], lr=1e-5)
    for i in range(1000):
        hypothesis = x.matmul(w) + b
        cost = torch.mean((hypothesis-y_train)**2)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        if i % 100 == 0:
            print('Epoch {:4d}/{} w: {}, b: {:.3f} Cost: {:.6f}'.format(
                i, 1000,
                w,
                b.item(),
                cost.item()
            ))


def nn_module():
    x_train = torch.FloatTensor([[1], [2], [3]])
    y_train = torch.FloatTensor([[2], [4], [6]])
    model = nn.Linear(1, 1)
    '''
    print(list(model.parameters()))
    [Parameter containing: tensor([[0.5153]], requires_grad=True), -> W 
    Parameter containing: tensor([-0.4414], requires_grad=True)] -> b

    model = nn.Linear(3, 1)
    [Parameter containing:tensor([[ 0.2975, -0.2548, -0.1119]], requires_grad=True), -> w
    Parameter containing:tensor([0.2710], requires_grad=True)] -> b
    '''
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    nb_epochs = 2000
    for epoch in range(nb_epochs+1):

        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)  # <== 파이토치에서 제공하는 평균 제곱 오차 함수

        # cost로 H(x) 개선하는 부분
        # gradient를 0으로 초기화
        optimizer.zero_grad()
        # 비용 함수를 미분하여 gradient 계산
        cost.backward()  # backward 연산
        # W와 b를 업데이트
        optimizer.step()

        if epoch % 100 == 0:
            # 100번마다 로그 출력
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))


if __name__ == '__main__':
    torch.manual_seed(1)
    nn_module()
