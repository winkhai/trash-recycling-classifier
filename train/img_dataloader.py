import torch
import torchvision
from torchvision import datasets, transforms
from img_aug import img_transforms

train_data_path = "../data/train/"
train_data = torchvision.datasets.ImageFolder(root=train_data_path,transform=img_transforms['train'])

valid_data_path = "../data/valid/"
valid_data = torchvision.datasets.ImageFolder(root=valid_data_path,transform=img_transforms['valid'])

test_data_path = "../data/test/"
test_data = torchvision.datasets.ImageFolder(root=test_data_path,transform=img_transforms['test'])

print(f'there are {len(train_data.classes)} classes in the training dataset.')
print(f'there are {len(valid_data.classes)} classes in the validation dataset.')
print(f'there are {len(test_data.classes)} classes in the testing dataset.')


#Params
batch_size = 64
num_workers = 0

train_loader = torch.utils.data.DataLoader(train_data,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           shuffle=False)

loaders = {
                                               'train': train_loader,
                                               'valid': valid_loader,
                                               'test': test_loader
                                           }

print(f'Num. of Images in Training Set: {len(train_loader.dataset)}')
print(f'Num. of Images in Validation Set: {len(valid_loader.dataset)}')
print(f'Num. of Images in Testing Set: {len(test_loader.dataset)}')
