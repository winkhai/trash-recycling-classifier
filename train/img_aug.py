from torchvision import datasets, transforms


std_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

img_transforms = {'train': transforms.Compose([transforms.RandomResizedCrop(size=256),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.RandomRotation(degrees=30),
                                               transforms.CenterCrop(size=224),
                                               transforms.ToTensor(),
                                               std_norm]),
                   'valid': transforms.Compose([transforms.Resize(size=256),
                                                transforms.CenterCrop(size=224),
                                                transforms.ToTensor(),
                                                std_norm]),
                   'test': transforms.Compose([transforms.Resize(size=256),
                                               transforms.CenterCrop(size=224),
                                               transforms.ToTensor(),
                                               std_norm])}
