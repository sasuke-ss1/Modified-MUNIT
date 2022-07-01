import torch 
import torch.nn
from torch.utils.data import DataLoader
from torchvision import transforms
from DA_net.data import ImageFolder
import yaml
from torchvision.utils import make_grid, save_image

def Folder_data(input_folder, batch_size, train, new_size=None,height=256, width=256, num_workers=4, crop=True):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5))]
    transform_list = [transforms.RandomCrop((height, width))] + transform_list if crop else transform_list
    transform_list = [transforms.Resize(new_size)] + transform_list if new_size is not None else transform_list
    transform_list = [transforms.RandomHorizontalFlip()] + transform_list if train else transform_list
    transform = transforms.Compose(transform_list)
    dataset = ImageFolder(input_folder, transform=transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=train, drop_last=True, num_workers=num_workers)
    return loader

def get_config(config):
    with open(config, 'r') as stream:
        return yaml.safe_load(stream)


def save_images(im1, im2, im3, im4, im5, iter):
    grid = make_grid([im1.cpu()[0], im2.cpu()[0], im3.cpu()[0], im4.cpu()[0], im5.cpu()[0]])
    save_image(grid, f"./checkpoints/S_images/{iter}.png")


