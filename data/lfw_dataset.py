import os
import torch
from utils.utils import download_resources
from utils.augmentations import resize_image, to_tensor
from PIL import Image
from sklearn.model_selection import train_test_split

def image_transforms(shape):
    def train_transforms(image):
        image = resize_image(image, shape=shape)
        # image = add_non_spatial_augmentations(image)
        image = to_tensor(image)

        return image

    return {'train': train_transforms}


class LFWDataset(torch.utils.data.Dataset):

    def __init__(self, base_folder, download=True, split_name: str = 'train'):
        super().__init__()
        self.base_folder = base_folder
        self.image_shape = (256, 256)
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

        self.X_train, x_val_test, self.Y_train, y_val_test = train_test_split(self.X, self.Y, test_size=0.3,
                                                                              random_state=42)
        self.X_val, self.X_test, self.Y_val, self.Y_test = train_test_split(x_val_test, y_val_test, test_size=1/3,
                                                                              random_state=42)

        if split_name == 'train':
            self.data = list(zip(self.X_train, self.Y_train))
        elif split_name == 'val':
            self.data = list(zip(self.X_val, self.Y_val))
        elif split_name == 'test':
            self.data = list(zip(self.X_test, self.Y_test))

        data_transform = image_transforms(shape=self.image_shape)
        self.data_transform = data_transform['train']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        filename_x, filename_y = self.data[idx]

        image = Image.open(filename_x)
        seg = Image.open(filename_y)


        if self.data_transform:
            image = self.data_transform(image)
            seg = self.data_transform(seg)


        sample = {'image': image, 'seg': seg}

        return sample
