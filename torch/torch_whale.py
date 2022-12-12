import torch


def torch_available_device():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def torch_manual_seed(seed):
    torch.manual_seed(seed)
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        torch.backends.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    else:
        pass

def xavier_uniform(layer):
    if hasattr(layer, 'weight'):
        print(f'apply xavier_uniform to {layer}')
        torch.nn.init.xavier_uniform_(layer.weight)

    
