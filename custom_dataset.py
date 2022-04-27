import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import glob
import numpy as np
import torchvision.transforms as transforms
import random
import cv2 as cv
from sklearn.model_selection import train_test_split
import torch
import torchvision
import torchdata as td

BATCH_SIZE = 4
NUM_WORKER = 3
IMAGE_SIZE = 128


class CougDataset(Dataset):
    def __init__(self, img_dir, transform):
        self.image_paths = self._create_paths(glob.glob(img_dir + "\\*"))
        self.classes = [self._get_class_from_image_path for path in self.image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv.imread(image_filepath)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        label = self._get_class_from_image_path(image_filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def _create_paths(self, all_paths):
        return [str(path) for path in all_paths]

    def _get_class_from_image_path(self, image_path):
        class_name = image_path.split("\\")[-1].split("_")[0]
        if class_name == "coug":
            return 1
        else:
            return 0


# #1.classes.append()
# # get all the paths from train_data_path and append image paths and class to to respective lists
# # eg. train path-> 'images/train/26.Pont_du_Gard/4321ee6695c23c7b.jpg'
# # eg. class -> 26.Pont_du_Gard
# for data_path in glob.glob(train_data_path + '/*'):
#     classes.append(data_path.split('/')[-1])
#     train_image_paths.append(glob.glob(data_path + '/*'))

# train_image_paths = list(flatten(train_image_paths))
# random.shuffle(train_image_paths)

# print('train_image_path example: ', train_image_paths[0])
# print('class example: ', classes[0])

# #2.
# # split train valid from train paths (80,20)
# train_image_paths, valid_image_paths = train_image_paths[:int(0.8*len(train_image_paths))], train_image_paths[int(0.8*len(train_image_paths)):]

# #3.
# # create the test_image_paths
# test_image_paths = []
# for data_path in glob.glob(test_data_path + '/*'):
#     test_image_paths.append(glob.glob(data_path + '/*'))

# test_image_paths = list(flatten(test_image_paths))

# print("Train size: {}\nValid size: {}\nTest size: {}".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))


def main():

    img_dir = "D:\\data\\magcougdataset\\cougnotcoug"

    # data_transform = torchvision.transforms.Compose(
    #     [
    #         torchvision.transforms.RandomResizedCrop(224),
    #         torchvision.transforms.RandomHorizontalFlip(),
    #         torchvision.transforms.ToTensor(),
    #         torchvision.transforms.Normalize(
    #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #         ),
    #     ]
    # )

    data_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(IMAGE_SIZE, max_size=IMAGE_SIZE * 2),
            transforms.RandomCrop(IMAGE_SIZE, pad_if_needed=True),
            transforms.ToTensor(),
            # torchvision.transforms.Normalize(
            #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            # ),
        ]
    )

    full_dataset = CougDataset(img_dir=img_dir, transform=data_transform)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, test_size]
    )

    print(f"train_size: {train_size}, test_size: {test_size}")

    print(
        "The shape of tensor for 50th image in train dataset: ",
        train_dataset[49][0].shape,
    )
    print("The label for 50th image in train dataset: ", train_dataset[49][1])

    # #######################################################
    # #                  Define Dataloaders
    # #######################################################

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
    )

    # print("loaders constructed")
    # train_iter = iter(train_loader)
    # print("iterator constructed")

    # batch = next(train_iter)
    # for i in range(BATCH_SIZE):
    #     plt.subplot(1, BATCH_SIZE, i+1)
    #     plt.title(batch[1][i])
    #     plt.imshow(batch[0][i].squeeze().permute(1, 2, 0))
    # plt.show()

if __name__ == "__main__":
    main()
