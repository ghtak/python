'''
Logistic Regression
-> Classification
 -> Sigmoid function : 1/(1 + e^-(Wx+b)))
  non convex graph

H(x) = sigmoid(W*x+b)

cost function
if y = 1 -> cost(H(x),y) = -log(H(x))
if y = 0 -> cost(H(x),y) = -log(1-H(x))

-> -( y log(H(x)) + (1-y) log(1-H(x) )

'''
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def sigmoid(x):
    return 1/(1+np.exp(-x))


def show_sigmoid():
    w = [0.5, 1, 2.0]
    w = [1, 1, 1]
    b = [0.5, 1, 1.5]
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = sigmoid(w[0]*x + b[0])
    y2 = sigmoid(w[1]*x + b[1])
    y3 = sigmoid(w[2]*x + b[2])

    plt.plot(x, y1, 'r', linestyle='--')  # W의 값이 0.5일때
    plt.plot(x, y2, 'g')  # W의 값이 1일때
    plt.plot(x, y3, 'b', linestyle='--')  # W의 값이 2일때
    plt.plot([0, 0], [1.0, 0.0], ':')  # 가운데 점선 추가
    plt.title('Sigmoid Function')
    plt.show()


def logistic_regression():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)
    '''
    W = torch.zeros((2,1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # hypothesis = 1/(1+torch.exp(-(x_train.matmul(W)+b))) 
    hypothesis = torch.sigmoid(x_train.matmul(W)+b)
    #  -(y log(H(x)) + (1-y) log(1-H(x) )
    # loss = -(y_train * torch.log(hypothesis) + (1-y_train) * torch.log(1-hypothesis))
    # cost = loss.mean()
    cost = F.binary_cross_entropy(hypothesis, y_train)
    print(hypothesis, cost)
    '''
    # 모델 초기화
    W = torch.zeros((2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    # optimizer 설정
    optimizer = optim.SGD([W, b], lr=1)

    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):

        # Cost 계산
        hypothesis = torch.sigmoid(x_train.matmul(W) + b)
        cost = (-(y_train * torch.log(hypothesis) +
                  (1 - y_train) * torch.log(1 - hypothesis))).mean()

        # cost로 H(x) 개선
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 100번마다 로그 출력
        if epoch % 100 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, cost.item()
            ))
    prediction = hypothesis >= torch.FloatTensor([0.5])
    print(prediction)


def raw_model():
    return nn.Sequential(
        nn.Linear(2, 1),  # input_dim = 2, output_dim = 1
        nn.Sigmoid()  # 출력은 시그모이드 함수를 거친다
    )

def class_model():
    class BinaryClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2,1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            return self.sigmoid(self.linear(x))
    return BinaryClassifier()


def nn_logistic_regression():
    x_data = [[1, 2], [2, 3], [3, 1], [4, 3], [5, 3], [6, 2]]
    y_data = [[0], [0], [0], [1], [1], [1]]
    x_train = torch.FloatTensor(x_data)
    y_train = torch.FloatTensor(y_data)
    model = class_model()
    optimizer = optim.SGD(model.parameters(), lr=1)

    nb_epochs = 1000
    for epoch in range(nb_epochs + 1):
        out = model(x_train)
        cost = F.binary_cross_entropy(out, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # 20번마다 로그 출력
        if epoch % 10 == 0:
            prediction = out >= torch.FloatTensor(
                [0.5])  # 예측값이 0.5를 넘으면 True로 간주
            correct_prediction = prediction.float() == y_train  # 실제값과 일치하는 경우만 True로 간주
            accuracy = correct_prediction.sum().item() / len(correct_prediction)  # 정확도를 계산
            print('Epoch {:4d}/{} Cost: {:.6f} Accuracy {:2.2f}%'.format(  # 각 에포크마다 정확도를 출력
                epoch, nb_epochs, cost.item(), accuracy * 100,
            ))


if __name__ == '__main__':
    torch.manual_seed(1)
    nn_logistic_regression()
