# CMU 16-726 Learning-Based Image Synthesis / Spring 2022, Assignment 3
# The code base is based on the great work from CSC 321, U Toronto
# https://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-code.zip

import glob
import os
import PIL.Image as Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataSet(Dataset):
    """Load images under folders"""
    def __init__(self, texture_dir='stylegan3/datasets/all_texture', person_dir='stylegan3/datasets/train', 
                 transform=None):
        # self.main_dir = main_dir
        self.transform = transform
        self.all_texture_imgs = glob.glob(os.path.join(texture_dir, '*.jpg'))
        self.all_person_imgs = glob.glob(os.path.join(person_dir, '*.png'))
        print(len(self))

    def __len__(self):
        return len(self.all_person_imgs)

    def __getitem__(self, idx):
        person_img_loc = self.all_person_imgs[idx]
        person_image = Image.open(person_img_loc).convert("RGB")
        texture_img_loc = self.all_texture_imgs[idx % len(self.all_texture_imgs)]
        texture_image = Image.open(texture_img_loc).convert("RGB")
        return self.transform(person_image), self.transform(texture_image)


def get_data_loader(texture_dir='stylegan3/datasets/all_texture', person_dir='stylegan3/datasets/train', opts=None):
    """Creates training and test data loaders.
    """
    if opts.data_preprocess == 'basic':
        train_transform = [
        transforms.Resize(opts.image_size, Image.BICUBIC),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    elif opts.data_preprocess == 'deluxe':
        # todo: add your code here: below are some ideas for your reference
        load_size = int(1.1 * opts.image_size)
        osize = [load_size, load_size]
        train_transform = [
            transforms.Resize(osize, Image.BICUBIC),
            # transforms.RandomCrop(opts.image_size),
            transforms.ToTensor(),
            # transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    dataset = CustomDataSet(texture_dir, person_dir, transforms.Compose(train_transform))
    dataloader = DataLoader(dataset=dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    return dataloader
