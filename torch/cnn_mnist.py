import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_whale import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from softmax_mnist import get_datasets

def conv_exam():
    conv1 = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
    pool = nn.MaxPool2d(2)
    print(conv1, conv2, pool)
    inputs = torch.Tensor(1, 1, 28, 28)
    out = conv1(inputs)
    print(out.size())
    out = pool(out)
    print(out.size())
    out = conv2(out)
    print(out.size())
    out = pool(out)
    print(out.size())
    out = out.view(out.size(0), -1)
    print(out.size())


class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Linear(7*7*64, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class CNN2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), # ?x28x28x1 -> ?x28x28x32
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # ?x28x28x32 -> ?x14x14x32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # ?x14x14x32 -> ?x14x14x64
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # ?x14x14x64 -> ?x7x7x64
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), #  ?x7x7x64 ->  ?x7x7x128
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1) # ?x7x7x128 -> ?x4x4x128
        )

        fc = nn.Linear(4*4*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(fc.weight)

        self.layer4 = nn.Sequential(
            fc, 
            nn.LeakyReLU(),
            nn.Dropout()
        )
        self.fc = nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc(out)
        return out


def mnist():
    device = torch_available_device()

    learning_rate = 0.001
    training_epochs = 15
    batch_size = 100

    mnist_train, mnist_test, data_loader = get_datasets(batch_size)
    model = CNN2().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = len(data_loader)
        for X,Y in data_loader:
            X = X.to(device)
            Y = Y.to(device)
            out = model(X)
            cost = criterion(out, Y)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            avg_cost += cost / total_batch

        print('[Epoch: {:>4}] cost = {:>.9}'.format(epoch + 1, avg_cost))

    with torch.no_grad():
        X_test = mnist_test.data.view(len(mnist_test), 1, 28, 28).float().to(device)
        Y_test = mnist_test.targets.to(device)
        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, 1) == Y_test
        accuracy = correct_prediction.float().mean()
        print('Accuracy:', accuracy.item())

if __name__ == '__main__':
    torch_manual_seed(81)
    mnist()


