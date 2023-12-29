import torch
from PIL import Image
from torchvision import transforms
def resize_image(image: Image, shape: tuple) -> Image:
    image_transform = transforms.Resize(shape, interpolation=Image.LANCZOS)
    image = image_transform(image)

    return image


def to_tensor(image: Image) -> torch.Tensor:
    transform = transforms.ToTensor()
    image = transform(image).type('torch.FloatTensor')
    return image
