import pickle

from torch.cuda.amp import autocast as autocast, GradScaler

def unpickle(file):
    
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


if __name__ == '__main__':
    dict = unpickle('cifar-10-batches-py\data_batch_1')
    print(len(dict))


