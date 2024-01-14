import numpy as np
import torch.cuda
import torch.nn as nn
import torch.optim as optim
import wandb
from data.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
from metrics.segmentation_metrics import compute_metrics
from models.uNet import UNet
from tqdm import tqdm
from model_checkpoint import ModelCheckpoint

step = 0


def test(model, test_loader, device):
    model.eval()
    mean_accuracy, mean_iou, mean_fw_iou = [], [], []

    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), desc="evaluate_unet"):
            imgs = data["image"]
            segs = data["seg"]
            imgs, segs = imgs.to(device), segs.to(device)

            segs_pred = model(imgs)

            mpa, m_iou = compute_metrics(segs, segs_pred)
            mean_accuracy.append(mpa)
            mean_iou.append(m_iou)

    wandb.log({"Test Mean Pixel Acc": np.mean(mean_accuracy),
               "Test Mean IoU": np.mean(mean_iou),
               "Test Frequency Weighted IoU": np.mean(mean_fw_iou)}, step=step)


def val(model, val_loader, criterion, config, device, epoch, model_ckpt):
    global step
    running_loss = 0.0
    mean_accuracy, mean_iou, mean_fw_iou = [], [], []
    table = wandb.Table(columns=["id", "image", "pred", "gt"])

    model.eval()

    pbar = tqdm(enumerate(val_loader, 0),
                unit=' images',
                unit_scale=config.batch_size,
                total=len(val_loader),
                smoothing=0,
                disable=False)

    with torch.no_grad():
        for (batch_idx, data) in pbar:
            imgs = data["image"]
            segs = data["seg"]
            imgs, segs = imgs.to(device), segs.to(device)

            segs_pred = model(imgs)
            loss = criterion(segs_pred, segs)

            if batch_idx < 5:
                table.add_data(
                    *[f'{step}_{batch_idx}', wandb.Image(imgs[0]), wandb.Image(segs_pred[0]), wandb.Image(segs[0])])

            running_loss += float(loss)
            val_loss = float(running_loss) / (batch_idx + 1)

            pbar.set_description(f'Validation [ E {epoch}, L {loss}, L_Avg {val_loss}')

            mpa, m_iou, m_fw_iou = compute_metrics(segs, segs_pred)
            mean_accuracy.append(mpa)
            mean_iou.append(m_iou)
            mean_fw_iou.append(m_fw_iou)

        val_loss = float(running_loss) / len(val_loader)

        wandb.log({"Validation Loss": val_loss,
                   "Validation Mean Pixel Acc": np.mean(mean_accuracy),
                   "Validation Mean IoU": np.mean(mean_iou),
                   "Validation Frequency Weighted IoU": np.mean(mean_fw_iou)}, step=step)

        wandb.log({"Images Data": table})

        model_ckpt(model, epoch, np.mean(mean_iou))


def train(model, train_loader, criterion, optimizer, config, device, epoch):
    running_loss = 0.0
    global step

    model.train()
    pbar = tqdm(enumerate(train_loader, 0),
                unit=' images',
                unit_scale=config.batch_size,
                total=len(train_loader),
                smoothing=0,
                disable=False)

    for (batch_idx, data) in pbar:

        imgs = data["image"]
        segs = data["seg"]
        imgs, segs = imgs.to(device), segs.to(device)

        optimizer.zero_grad()
        segs_pred = model(imgs)
        loss = criterion(segs_pred, segs)

        loss.backward()
        optimizer.step()

        running_loss += float(loss)
        step += len(imgs)
        train_loss = float(running_loss) / (batch_idx + 1)
        pbar.set_description(f'Training [ E {epoch}, L {loss}, L_Avg {train_loss}')

        batch_idx += 1
        if batch_idx % config.log_freq == 0:
            wandb.log({"Training Loss": train_loss}, step=step)


def model_pipeline(hyperparameters=None):
    with wandb.init(project="pytorch-demo", config=hyperparameters, dir=rf"C:\work\an 3\dl\face-segmentation"):
        config = wandb.config

        device = "cuda" if torch.cuda.is_available() else "cpu"

        train_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="train")
        val_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="val")
        test_dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="test")

        train_loader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  pin_memory=True,
                                  shuffle=False,
                                  sampler=None,
                                  num_workers=0)

        val_loader = DataLoader(val_dataset,
                                batch_size=config.batch_size,
                                pin_memory=True,
                                shuffle=False,
                                sampler=None,
                                num_workers=0)

        test_loader = DataLoader(test_dataset,
                                 batch_size=config.batch_size,
                                 pin_memory=True,
                                 shuffle=False,
                                 sampler=None,
                                 num_workers=0)

        model = UNet(n_channels=3, n_classes=3)
        model = model.to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        model_checkpoint = ModelCheckpoint(0.0, True, 5, "mean_iou")
        for epoch in range(config.epochs):
            val(model, val_loader, criterion, config, device, epoch, model_checkpoint)
            train(model, train_loader, criterion, optimizer, config, device, epoch)

        test(model, test_loader, device)
    wandb.finish()


if __name__ == '__main__':
    config = dict(
        epochs=30,
        classes=3,
        batch_size=8,
        learning_rate=0.0001,
        dataset="LFW",
        architecture="UNet",
        log_freq=10)

    model_pipeline(config)
