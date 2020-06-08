from torch.utils.data import Dataset, DataLoader
import pickle
import torchvision.transforms as transforms
import os


class Cifar100(Dataset):
    def __init__(self, root, train=False):
        self.cifar_train_mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
        self.cifar_train_std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

        self.path = os.path.join(root, 'test')
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(self.cifar_train_mean, self.cifar_train_std)])
        if train:
            self.path = os.path.join(root, 'train')
            self.transform = transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.RandomHorizontalFlip(),
                 transforms.RandomCrop(32, padding=4),
                 transforms.RandomRotation(15),
                 transforms.ToTensor(),
                 transforms.Normalize(self.cifar_train_mean, self.cifar_train_std)])

        with open(self.path, 'rb') as cifar_train:
            self.cifar = pickle.load(cifar_train, encoding='bytes')

        self.labels = self.cifar['fine_labels'.encode()]
        self.data = self.cifar['data'.encode()]
        self.data = self.data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index]
        image = self.transform(self.data[index])
        return label, image


def Cifar_loader(path, batch_size=64,
                 shuffle=True, train=False,
                 num_workers=None):

    data = Cifar100(path, train)
    loader = DataLoader(data, shuffle=shuffle,
                        batch_size=batch_size,
                        num_workers=num_workers)
    return loader
