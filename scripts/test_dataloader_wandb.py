import wandb
from data.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


if __name__ == '__main__':
    wandb.init(
        project="face-segmentation",
        dir=rf"C:\work\an 3\dl\face-segmentation"
    )

    dataset = LFWDataset(download=False, base_folder='lfw_dataset', split_name="test")

    train_loader = DataLoader(dataset,
                                    batch_size=8,
                                    pin_memory=True,
                                    shuffle=False,
                                    sampler=None,
                                    num_workers=0)

    table = wandb.Table(columns=["image", "pred"])
    for batch_idx, batch in enumerate(train_loader):
        grid = make_grid(batch['image'], nrow=4)
        grid_y = make_grid(batch['seg'], nrow=4)
        table.add_data(*[wandb.Image(grid), wandb.Image(grid_y)])
    wandb.log({"predictions_table": table}, commit=False)

    wandb.finish()