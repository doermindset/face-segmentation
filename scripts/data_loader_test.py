from torch.utils.tensorboard import SummaryWriter
from data.lfw_dataset import LFWDataset
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

if __name__ == '__main__':

    dataset = LFWDataset(download=False, base_folder='lfw_dataset')

    train_loader = DataLoader(dataset,
                                batch_size=8,
                                pin_memory=True,
                                shuffle=False,
                              sampler=None,
                                num_workers=0)
    summary = SummaryWriter('runs')

    for batch_idx, batch in enumerate(train_loader):
        print(batch_idx)
        grid = make_grid(batch['image'], nrow=4)
        summary.add_image(f"Images", grid, global_step=batch_idx)



