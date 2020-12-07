# Want to load:
# - Source image
# - Target image

# Transformations:
# - Rescale input images (3, 224, 224)
# - Dataset normalization
# - Randomly rotate images

import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

def dataloader(root = "\clean_dataset\train_data",
               image_size = 224,
               num_channels = 3,
               batch_size = 4,
               num_workers = 6):
#                mean = None,
#                std = None):

#     if (mean == None or std == None):
#         normset = torchvision.datasets.ImageFolder(root=root, transform=transforms.Compose([transforms.ToTensor()]))
#         normloader = torch.utils.data.DataLoader(normset, batch_size=4, shuffle=True, num_workers=2)

#         num_images = 0
#         mean = [0, 0, 0]
#         std = [0, 0, 0]
#         for inputs, _ in normloader:
#             num_images = num_images + 1
#             for i in range(3):
#                 mean[i] += inputs[:, i, :, :].mean()
#                 std[i] += inputs[:, i, :, :].std()
#         for i in range(3):
#             mean[i] = mean[i] / num_images
#             std[i] = std[i] / num_images
#         print(mean)
#         print(std)

    # transform = transforms.Compose([transforms.Normalize((mean[0], mean[1], mean[2]), (std[0], std[1], std[2]))],
    #                                transforms.Resize(image_size),
    #                                transforms.RandomRotation(45),
    #                                transforms.ToTensor())
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomRotation(45),
        transforms.ToTensor()
    ])

    image_data = dset.ImageFolder(root=root,
                                  transform=transform)

    dataset = torch.utils.data.TensorDataset(image_data, image_data)

    dataloader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers)

    return dataloader
