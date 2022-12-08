import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random


def set_seed():
    torch.manual_seed(81)
    random.seed(81)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(81)


def get_datasets():
    # MNIST
    mnist_train = dsets.MNIST(root='MNIST_data/',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)

    mnist_test = dsets.MNIST(root='MNIST_data/',
                             train=False,
                             transform=transforms.ToTensor(),
                             download=True)

    data_loader = DataLoader(dataset=mnist_train,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True)
    return mnist_train, mnist_test, data_loader

if __name__ == '__main__':
    set_seed()

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    nb_epochs = 15
    batch_size = 100

    mnist_train, mnist_test, data_loader = get_datasets()

    linear = nn.Linear(784,10, bias=True).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

    for epoch in range(nb_epochs):
        avg_cost = 0
        total_batch = len(data_loader)
        for X,Y in data_loader:
            X = X.view(-1,784).to(device)
            Y = Y.to(device)

            pred = linear(X)
            cost = criterion(pred, Y)

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
    print('Learning finished')

    with torch.no_grad():
        X_test = mnist_test.test_data.view(-1,784).float().to(device)
        Y_test = mnist_test.test_labels.to(device)
        prediction = linear(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())
        # MNIST 테스트 데이터에서 무작위로 하나를 뽑아서 예측을 해본다
        r = random.randint(0, len(mnist_test) - 1)
        X_single_data = mnist_test.test_data[r:r + 1].view(-1, 28 * 28).float().to(device)
        Y_single_data = mnist_test.test_labels[r:r + 1].to(device)

        print('Label: ', Y_single_data.item())
        single_prediction = linear(X_single_data)
        print('Prediction: ', torch.argmax(single_prediction, 1).item())

        plt.imshow(mnist_test.test_data[r:r + 1].view(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()
