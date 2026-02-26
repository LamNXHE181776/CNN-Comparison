import torch
from torchvision import transforms 
from torch.utils import data

import copy

import matplotlib.pyplot as plt
import seaborn as sns

def preview_image(cols, rows, dataset):
    fig, axs = plt.subplots(cols, rows, figsize = (7, 7))
    axs = axs.flatten()

    class_names = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot"
    }


    for ax in axs:
        sample_idx = torch.randint(len(dataset), (1,)).item()
        img, label = dataset[sample_idx]
        ax.axis("off")
        ax.set_title(class_names[label],)
        ax.imshow(img, cmap = "grey")
    plt.tight_layout()
    plt.show()



def data_transform(dataset_full, dataset_test, batch_size, RATIO = 0.85, use_augmentation=True):
    n_train_examples = int(len(dataset_full) * RATIO)
    n_val_examples = len(dataset_full) - n_train_examples


    mean = dataset_full.data.float().mean() / 255
    std  = dataset_full.data.float().std() / 255

    # Define transforms
    base_train_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ]

    if use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(28, padding=1),
            transforms.RandomAffine(degrees=6, translate=(0.05, 0.05), scale=(0.97, 1.03)),
            transforms.RandomHorizontalFlip(p=0.5),
            *base_train_transforms,
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        train_transform = transforms.Compose(base_train_transforms)

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[mean], std=[std])
    ])


    dataset_full_train = copy.deepcopy(dataset_full)
    dataset_full_val = copy.deepcopy(dataset_full)


    dataset_full_train.transform = train_transform
    dataset_full_val.transform = test_transform
    dataset_test.transform = test_transform

    train_data_indice, val_data_indice = data.random_split(range(len(dataset_full)), 
                                            [n_train_examples, n_val_examples], 
                                            generator=torch.Generator().manual_seed(42))

    train_data = data.Subset(dataset_full_train, train_data_indice)
    val_data = data.Subset(dataset_full_val, val_data_indice)

    train_dataloader = data.DataLoader(train_data, 
                                    batch_size=batch_size, 
                                    shuffle=True,
                                    num_workers=3,
                                    pin_memory=True,
                                    persistent_workers=True,
                                    drop_last=True)

    val_dataloader = data.DataLoader(val_data, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=3,
                                    pin_memory=True,
                                    persistent_workers=True)

    test_dataloader = data.DataLoader(dataset_test, 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=3,
                                    pin_memory=True,
                                    persistent_workers=True)
    
    return train_dataloader, val_dataloader, test_dataloader

