import torch.cuda
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from data.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
from models.uNet import UNet
from utils.config import Config
from utils.utils import read_params, sample_to_cuda
from tqdm import tqdm





def train(config, train_loader, model, optimizer, epoch, device, summary, criterion):
    for batch_idx, batch in enumerate(train_loader):
        model.train()

        n_train_batches = len(train_loader)

        pbar = tqdm(enumerate(train_loader, 0),
                    unit=' images',
                    unit_scale=config.dataset.batch_size,
                    total=len(train_loader),
                    smoothing=0,
                    disable=False)
        running_loss = running_recall = grad_norm_disp = grad_norm_pose = grad_norm_keypoint = 0.0
        train_progress = float(epoch) / float(config.epochs)

        log_freq = 10

        for (i, data) in pbar:

            # calculate loss
            optimizer.zero_grad()
            data_cuda = sample_to_cuda(data)
            imgs = data_cuda["image"]
            segs = data_cuda["seg"]
            segs_pred = model(imgs)

            loss = criterion(segs_pred, segs)
            # compute gradient
            loss.backward()

            running_loss += float(loss)

            # SGD step
            optimizer.step()
            # pretty progress bar

            i += 1
            if i % log_freq == 0:
                with torch.no_grad():
                    if summary:
                        train_metrics = {
                            'train_loss': running_loss / (i + 1),
                            'train_progress': train_progress,
                        }

                        for param_group in optimizer.param_groups:
                            train_metrics['learning_rate'] = param_group['lr']

                        for k, v in train_metrics.items():
                            summary.add_scalar(k, v, i)


if __name__ == '__main__':

    params = read_params(rf"C:\work\an 3\dl\face-segmentation\config.yaml")
    config = Config(params)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="train")

    train_loader = DataLoader(train_dataset,
                                batch_size=8,
                                pin_memory=True,
                                shuffle=False,
                                sampler=None,
                                num_workers=0)

    model = UNet(n_channels=3, n_classes=3)
    model = model.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.learning_rate)

    summary = SummaryWriter(rf'C:\work\an 3\dl\face-segmentation\runs')

    criterion = nn.CrossEntropyLoss()

    for epoch in range(config.train.epochs):
        train(config.train, train_loader, model, optimizer, epoch, device, summary, criterion)



