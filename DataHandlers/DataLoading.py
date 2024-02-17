from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Generator.SettingsEM import GENERATED_FILE_PATH
import torch


def loadOriginalTrainSet():
    # Path where the dataset will be downloaded
    DATASET_PATH = "./Saved"

    # Transformations applied on each image => make them a tensor and normalize between -1 and 1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    # Loading the training MNIST. We need to split it into a training and validation part
    train_set = datasets.MNIST(
        root=DATASET_PATH, train=True, transform=transform, download=True
    )
    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    return train_loader


def loadTestSet():
    # Path where the dataset will be downloaded
    DATASET_PATH = "./Saved"

    # Transformations applied on each image => make them a tensor and normalize between -1 and 1
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    test_set = datasets.MNIST(
        root=DATASET_PATH, train=False, transform=transform, download=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )
    return test_loader


def loadGeneratedSet():
    generated_dataset = torch.load(GENERATED_FILE_PATH)
    generated_loader = DataLoader(
        generated_dataset,
        batch_size=128,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    return generated_loader


# def addGeneratedData(train_loader: DataLoader):
#     generated_dataset = torch.load(GENERATED_FILE_PATH)

#     images = []
#     labels = []
#     for image, label in generated_dataset:
#         images.append(image)
#         labels.append(label)

#     for image, label in train_loader.dataset:
#         images.append(image)
#         labels.append(label)

#     combined_dataset = CustomMNISTDataset(images, labels)
#     combined_loader = DataLoader(
#         combined_dataset,
#         batch_size=128,
#         shuffle=True,
#         drop_last=True,
#         num_workers=4,
#         pin_memory=True,
#         persistent_workers=True,
#     )

#     showImages(generated_dataset, "Generated Images")
#     showImages(train_loader.dataset, "Original Images")
#     return combined_loader
