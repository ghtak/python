import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.x_data = [[73,  80,  75],
                       [93,  88,  93],
                       [89,  91,  90],
                       [96,  98,  100],
                       [73,  66,  70]]
        self.y_data = [[152],  [185],  [180],  [196],  [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])
        return x, y


def make_tensordataset():
    x_train = torch.FloatTensor([[73,  80,  75],
                                 [93,  88,  93],
                                 [89,  91,  90],
                                 [96,  98,  100],
                                 [73,  66,  70]])
    y_train = torch.FloatTensor([[152],  [185],  [180],  [196],  [142]])
    dataset = TensorDataset(x_train, y_train)
    return dataset


def make_customdataset():
    return CustomDataset()


def dataset_test():
    dataloader = DataLoader(make_customdataset(), batch_size=2, shuffle=True)
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
    nb_epochs = 20
    for epoch in range(nb_epochs + 1):
        for batch_index, samples in enumerate(dataloader):
            x_train, y_train = samples
            prediction = model(x_train)
            cost = F.mse_loss(prediction, y_train)
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, nb_epochs, batch_index+1, len(dataloader),
                cost.item()
            ))

    new_var = torch.FloatTensor([[73, 80, 75]])
    pred = model(new_var)
    print(f"estimate : {pred}")


if __name__ == '__main__':
    torch.manual_seed(1)
    dataset_test()
