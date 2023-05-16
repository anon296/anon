import numpy as np
from os.path import join
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.model_selection import train_test_split
from torchvision.transforms import Compose, ToTensor
from dl_utils.torch_misc import CifarLikeDataset

from .fast_tensor_dataloader import FastTensorDataLoader


def get_celeba_dataloader(batch_size=128, path_to_data='../celeba_64', warm_up=True):
    data_transforms = transforms.Compose([
        # transforms.Resize(64),
        # transforms.CenterCrop(64),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_kwargs = {
        'target_type': 'attr',
        'transform': data_transforms,
    }
    import pdb; pdb.set_trace()  # XXX BREAKPOINT
    train_data = datasets.CelebA(path_to_data, split='train', download=True, **dataset_kwargs)
    test_data = datasets.CelebA(path_to_data, split='test', **dataset_kwargs)
    # warm_up_data = WarmUpCelebADataset(path_to_data, split='train', target_transform=target_transforms, **dataset_kwargs)

    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'pin_memory': True,
        # 'pin_memory': False,
        'num_workers': 4,
    }
    train_loader = DataLoader(train_data, **dataloader_kwargs)
    test_loader = DataLoader(test_data, **dataloader_kwargs)
    # warm_up_loader = DataLoader(warm_up_data, **dataloader_kwargs)

    warm_up_loader = None
    if warm_up:
        # target_transforms = transforms.Compose([
        #     lambda x: x[celeba_good_columns],
        #     # lambda x: torch.flatten(F.one_hot(x, num_classes=2))
        #     my_celeba_target_transfrom
        # ])
        warm_up_x, warm_up_y = WarmUpCelebADataset(path_to_data, count=800, **dataset_kwargs).get_tensors()  # TODO: If it is good, make the class simpler
        warm_up_loader = FastTensorDataLoader(warm_up_x, warm_up_y, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, warm_up_loader


def get_maybe_zs_dataloader(test_level, dset_name, exclude_combo, exclude_combo_vals, batch_size=128, fraction=1.):
    dirpath = 'datasets'
    if test_level == 2:
        archive = np.load(join(dirpath,f'{dset_name}_small.npz'))
        warmuploader_size = 100
    elif test_level == 1:
        archive = np.load(join(dirpath,f'{dset_name}_medium.npz'))
        warmuploader_size = 500
    else:
        archive = np.load(join(dirpath,f'{dset_name}.npz'))
        warmuploader_size = 3000
    imgs = archive['imgs']
    class_labels = archive['latents_classes']
    #class_labels = class_labels[:,np.array([4,3,1,2,0])] # Have reordered in the dset so no longer need this
    if exclude_combo == [-1,-1]:
        train_set_imgs, test_set_imgs, train_set_labels, test_set_labels = train_test_split(imgs,class_labels,test_size=0.2)
    else:
        m1 = class_labels[:,exclude_combo[0]] == exclude_combo_vals[0]
        m2 = class_labels[:,exclude_combo[1]] == exclude_combo_vals[1]
        test_set_mask = np.logical_and(m1,m2)
        train_set_imgs, train_set_labels = imgs[~test_set_mask], class_labels[~test_set_mask]
        test_set_imgs, test_set_labels = imgs[test_set_mask], class_labels[test_set_mask]
    to_float = lambda t: t.float()
    transform = Compose([ToTensor(),to_float])
    trainset = CifarLikeDataset(train_set_imgs,train_set_labels,transform)
    idx = np.random.choice(len(trainset),size=warmuploader_size,replace=False)
    warmup_imgs, warmup_labels = train_set_imgs[idx], train_set_labels[idx]
    warmupset = CifarLikeDataset(warmup_imgs,warmup_labels,transform)
    testset = CifarLikeDataset(test_set_imgs,test_set_labels,transform)
    trainloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        #'pin_memory': device.type != 'cpu',
        'pin_memory': True,
        #'num_workers': 0 if device.type == 'cpu' else 4,
        'num_workers': 4,
    }
    trainloader = DataLoader(trainset, **trainloader_kwargs)
    warmuploader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        #'pin_memory': device.type != 'cpu',
        'pin_memory': True,
        'num_workers': 4,
    }
    warmuploader = DataLoader(warmupset, **warmuploader_kwargs)
    testloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        #'pin_memory': device.type != 'cpu',
        'pin_memory': True,
        'num_workers': 4,
    }
    testloader = DataLoader(testset, **testloader_kwargs)

    return trainloader, testloader, warmuploader

class DSpritesDataset(Dataset):
    # Color[1], Shape[3], Scale, Orientation, PosX, PosY
    def __init__(self, path_to_data, fraction=1.):
        data = np.load(path_to_data)
        self.imgs = data['imgs']
        self.imgs = np.expand_dims(self.imgs, axis=1)
        self.classes = data['latents_classes']
        if fraction < 1:
            indices = np.random.choice(737280, size=int(fraction * 737280), replace=False)
            self.imgs = self.imgs[indices]
            self.classes = self.classes[indices]
        # self.attrs = data['latents_values'][indices]
        # self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        # # Each image in the dataset has binary values so multiply by 255 to get
        # # pixel values
        # sample = self.imgs[idx] * 255
        # # Add extra dimension to turn shape into (H, W) -> (H, W, C)
        # sample = sample.reshape(sample.shape + (1,))

        # if self.transform:
        #     sample = self.transform(sample)
        # Since there are no labels, we just return 0 for the "label" here
        # return sample, (self.classes[idx], self.attrs[idx])

         return torch.tensor(self.imgs[idx], dtype=torch.float, device='cuda'), torch.tensor(self.classes[idx], device='cuda')

    def get_tensors(self):
        return torch.tensor(self.imgs, dtype=torch.float).cuda(), torch.tensor(self.classes).cuda()
