import torch
import torch.nn as nn
from torch_whale import *
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits


def xor_perceptron():
    device = torch_available_device()
    # xor
    X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
    Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)
    #model = nn.Sequential(nn.Linear(2,1), nn.Sigmoid()).to(device)

    model = nn.Sequential(
        nn.Linear(2, 2),
        nn.Sigmoid(),
        nn.Linear(2, 1),
        nn.Sigmoid()
    ).to(device)
    criterion = nn.BCELoss().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.5)
    for step in range(10001):
        out = model(X)
        cost = criterion(out, Y)
        optim.zero_grad()
        cost.backward()
        optim.step()
        if step % 100 == 0:  # 100번째 에포크마다 비용 출력
            print(step, cost.item())

    with torch.no_grad():
        out = model(X)
        predicted = (out > 0.5).float()
        accuracy = (predicted == Y).float().mean()
        print('모델의 출력값(Hypothesis): ', out.detach().cpu().numpy())
        print('모델의 예측값(Predicted): ', predicted.detach().cpu().numpy())
        print('실제값(Y): ', Y.cpu().numpy())
        print('정확도(Accuracy): ', accuracy.item())


def multiple_layer_perceptron():
    device = torch_available_device()
    digits = load_digits()
    X = torch.FloatTensor(digits.data).to(device)
    Y = torch.LongTensor(digits.target).to(device)
    '''
    linear1 = nn.Linear(64, 32) # input_layer = 64, hidden_layer1 = 32
    linear2 = nn.Linear(32, 16) # hidden_layer2 = 32, hidden_layer3 = 16
    linear3 = nn.Linear(16, 10) # hidden_layer3 = 16, output_layer = 10
    
    torch.nn.init.xavier_uniform_(linear1.weight)
    torch.nn.init.xavier_uniform_(linear2.weight)
    torch.nn.init.xavier_uniform_(linear3.weight)

    model = nn.Sequential(
        linear1,
        nn.ReLU(),
        linear2,
        nn.ReLU(),
        linear3,
    ).to(device)
    '''

    model = nn.Sequential(
        nn.Linear(64, 32),  # input_layer = 64, hidden_layer1 = 32
        nn.ReLU(),
        nn.Linear(32, 16),  # hidden_layer2 = 32, hidden_layer3 = 16
        nn.ReLU(),
        nn.Linear(16, 10)  # hidden_layer3 = 16, output_layer = 10
    ).to(device)

    # model.apply(xavier_uniform)
    '''
    for layer in model.named_parameters(): 
        # layer ->  (param_name, param_weight)
        if 'weight' in layer[0] and 'layerNorm' not in layer[0]:
            print(layer)
            torch.nn.init.xavier_uniform_(layer[1])
    '''

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    losses = []
    for epoch in range(100):
        pred = model(X)
        loss = loss_fn(pred, Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print('Epoch {:4d}/{} Cost: {:.6f}'.format(
                epoch, 100, loss.item()
            ))

        losses.append(loss.item())

    plt.plot(losses)
    plt.show()


if __name__ == '__main__':
    torch_manual_seed(81)
    multiple_layer_perceptron()
