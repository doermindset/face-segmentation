import wandb
import torch
import os


class ModelCheckpoint:

    def __init__(self, best_metric_val, increasing_metric, n, metric_name):

        self.best_metric_val = best_metric_val
        self.increasing_metric = increasing_metric
        self.n = n
        self.saved_models = []
        self.metric_name = metric_name

    def __call__(self, model, epoch, metric_val):

        must_save = metric_val > self.best_metric_val if self.increasing_metric else metric_val < self.best_metric_val
        if must_save:
            self.best_metric_val = metric_val
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_metric_val': self.best_metric_val,
            }
            model_path = rf'checkpoints/{wandb.run.name}_ckpt_epoch_{epoch}_{self.metric_name}.pth'
            torch.save(checkpoint, model_path)

            self.write_artifact(model_path, metric_val)

            self.saved_models.append(model_path)

        if len(self.saved_models) > self.n:
            to_remove = self.saved_models.pop()
            os.remove(to_remove)

    def write_artifact(self, model_path, metric_val):

        artifact = wandb.Artifact(f"{wandb.run.name}_unet_m_iou", type='model', metadata={'metric': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)
