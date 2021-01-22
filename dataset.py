import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision import datasets


class MNIST_Dataset:

    def __init__(self, batch_size=4, dataset_path="", device=None, val=0):
        self.batch_size = batch_size
        self.device = device
        self.dataset_path = dataset_path
        self.x_mean, self.x_std = 0.5, 0.5#self.get_train_statistics()

        self.transform = self.get_transforms()
        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(val)

    def get_transforms(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ]
        transform = transforms.Compose(transform_list)
        return transform
    
    def get_train_statistics(self):
        train_dataset = datasets.MNIST(root=self.dataset_path, train=True)
        train_x = np.zeros((len(train_dataset), 32, 32))
        for i, (img, _) in enumerate(train_dataset):
            train_x[i] = img
        
        x_mean = np.mean(train_x)
        x_std = np.std(train_x)
        return x_mean, x_std

    def get_loaders(self, val=0):
        train_set = datasets.MNIST(root=self.dataset_path, train=True, transform=self.transform)
        

        if val > 0:
            train_size = int((1-val)*len(train_set))
            val_size = len(train_set) - train_size
            train_set, val_set = torch.utils.data.random_split(train_set, [train_size, val_size])

            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        else:
            val_loader = None

        train_loader = torch.utils.data.DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        test_set = datasets.MNIST(root=self.dataset_path, train=False, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader, test_loader

    def plot_image(self, image, label, filename="digit_image.png"):
        #image = np.transpose(image.numpy())
        image = image * self.x_std + self.x_mean # un-normalize
        plt.title(label)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.show()
        plt.savefig(filename)

class CIFAR10_Dataset:

    def __init__(self, batch_size=4, dataset_path="", device=None, val=0):
        self.batch_size = batch_size
        self.device = device
        self.dataset_path = dataset_path

        self.x_mean = (0.5, 0.5, 0.5)
        self.x_std = (0.5, 0.5, 0.5)

        self.transform, self.train_transform = self.get_transforms()

        self.train_loader, self.val_loader, self.test_loader = self.get_loaders(val)


    def get_transforms(self):
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize(self.x_mean, self.x_std)
        ]
        transform = transforms.Compose(transform_list)

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.x_mean, self.x_std),
            transforms.Pad(4),
            transforms.RandomCrop((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            ])

        return transform, train_transform

    def get_loaders(self, val=0):
        self.train_set = datasets.CIFAR10(root=self.dataset_path, train=True, transform=self.train_transform)
        

        if val > 1:
            train_size = len(self.train_set) - val
            val_size = val
            self.train_set, val_set = torch.utils.data.random_split(self.train_set, [train_size, val_size])

            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        elif val > 0:
            train_size = int((1-val)*len(self.train_set))
            val_size = len(self.train_set) - train_size
            self.train_set, val_set = torch.utils.data.random_split(self.train_set, [train_size, val_size])

            val_loader = torch.utils.data.DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        else:
            val_loader = None

        train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

        test_set = datasets.CIFAR10(root=self.dataset_path, train=False, transform=self.transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader

    def plot_image(self, image, label, filename="cifar_image.png"):
        #image = np.transpose(image.numpy())
        image = image * self.x_std + self.x_mean # un-normalize
        plt.title(label)
        plt.imshow(np.transpose(image.numpy(), (1, 2, 0)))
        plt.show()
        plt.savefig(filename)

if __name__ == "__main__":
    ds = MNIST_Dataset()
    
    print(ds.x_mean, ds.x_std)
    images, labels = iter(ds.train_loader).next()
    print(images.shape)
    ds.plot_image(torchvision.utils.make_grid(images), "...")
    