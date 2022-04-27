import torch
import torchvision
import torchvision.transforms as transforms

# custom dataset loading
import torch
import torchvision
import torchdata as td
import os
import PIL
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from skimage import io, transform
import pandas as pd
from torchvision.io import read_image
from skimage import data, io
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from torchvision.utils import save_image


# Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

# functions to show an image
def imshow(img):
    _img = np.squeeze(img)
    plt.imshow(_img)
    plt.show()


def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


def show_batch(dl, nmax=64):
    for images in dl:
        show_images(images, nmax)
        break


class CougData(Dataset):
    def __init__(
        self, annotations_file, img_dir, transform=None, target_transform=None
    ):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        image = io.imread(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


BATCH_SIZE = 64
NUM_WORKER = 3
IMAGE_SIZE = 64

dataset = CougData(
    annotations_file="D:\\data\\magcougdataset\\labels.csv",
    img_dir="D:\\data\\magcougdataset\\cougnotcoug\\",
)

total_count = len(dataset)
data_transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.RandomResizedCrop(224),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

train_count = int(0.7 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count)
)


print(len(train_dataset), len(test_dataset), len(valid_dataset))


# create the test/train/validation sets
trainloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
validloader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
testloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
dataloaders = {
    "train": trainloader,
    "val": validloader,
    "test": testloader,
}

classes = ("notcoug", "coug")


def main():
    img, label = dataset[3000]
    # imshow(img)
    print(classes[label])

    show_batch(trainloader, nmax=1)


if __name__ == "__main__":
    main()
