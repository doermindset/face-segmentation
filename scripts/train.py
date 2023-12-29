import torch.cuda
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
from models.uNet import UNet
from utils.config import Config
from utils.utils import read_params


if __name__ == '__main__':

    params = read_params("config.yaml")
    config = Config(params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = LFWDataset(download=False, base_folder='lfw_dataset')

    train_loader = DataLoader(dataset,
                                batch_size=8,
                                pin_memory=True,
                                shuffle=False,
                                sampler=None,
                                num_workers=0)

    model = UNet(n_channels=3, n_classes=3)
    model = model.to(device)
    optimizer = optim.Adam()

    summary = SummaryWriter('runs')

    for epoch in range(config.epochs):
        train(config, train_loader, model, optimizer, epoch, device, summary)

