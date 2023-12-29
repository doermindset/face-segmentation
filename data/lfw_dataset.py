import os
import torch
from utils.utils import download_resources
from utils.augmentations import resize_image, to_tensor
from PIL import Image


def image_transforms(shape):
    def train_transforms(image):
        image = resize_image(image, shape=shape)
        # image = crop_image(image)
        # image = add_non_spatial_augmentations(image)
        image = to_tensor(image)

        return image

    return {'train': train_transforms}


class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, base_folder, download=True, split_name: str = 'train'):
        super().__init__()
        self.base_folder = base_folder
        self.image_shape = (224, 224)
        if download:
            download_resources(base_folder)

        x_path = rf'C:\work\an 3\dl\face-segmentation\data\lfw_dataset\lfw_funneled'
        y_path = rf'C:\work\an 3\dl\face-segmentation\data\lfw_dataset\parts_lfw_funneled_gt_images'
        self.X = []
        self.Y = [os.path.join(y_path, img) for img in os.listdir(y_path) if
                  img.endswith('ppm') and not img.startswith('.')]
        for y in self.Y:
            y_basename = os.path.basename(y)
            self.X.append(os.path.join(x_path, y_basename[:-9] + "/" + y_basename.replace('.ppm', '.jpg')))

        data_transform = image_transforms(shape=self.image_shape)
        self.data_transform = data_transform['train']

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        filename = self.X[idx]

        image = Image.open(filename)

        if self.data_transform:
            image = self.data_transform(image)

        sample = {'image': image, 'y': self.Y[idx]}

        return sample
