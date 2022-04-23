import torch
import torchvision
from cougnotcoug import CougData
import torchdata as td
labels_path =  "D:\\data\\magcougdataset\\labels.csv"
data_path = "D:\\data\\magcougdataset\\cougnotcoug\\"
dataset = CougData(labels_path, data_path)
total_count = len(dataset)
BATCH_SIZE = 32
NUM_WORKER = 1

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

# # Single change, makes an instance of torchdata.Dataset
# # Works just like PyTorch's torch.utils.data.Dataset, but has
# # additional capabilities like .map, cache etc., see project's description
# model_dataset = td.datasets.WrapDataset(torchvision.datasets.ImageFolder(root))
# # Also you shouldn't use transforms here but below
train_count = int(0.7 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    dataset, (train_count, valid_count, test_count)
)

print(len(train_dataset), len(test_dataset), len(valid_dataset))
# # Apply transformations here only for train dataset

# train_dataset = train_dataset.map(data_transform)

# # Rest of the code goes the same

train_dataset_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
valid_dataset_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKER
)
test_dataset_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKER
)
dataloaders = {
    "train": train_dataset_loader,
    "val": valid_dataset_loader,
    "test": test_dataset_loader,
}