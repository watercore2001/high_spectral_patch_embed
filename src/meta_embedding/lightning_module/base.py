import dataclasses
from pytorch_lightning import LightningModule
from torch import nn
import torch
from torch import optim
from .lr_scheduler import CosineAnnealingWithWarmup
from .metrics import generate_classification_metric, separate_classes_metric

__all__ = ["AdamWCosineOptimArgs", "BaseModule"]

@dataclasses.dataclass
class AdamWCosineOptimArgs:
    weight_decay: float
    warmup_epochs: int
    annealing_epochs: int
    max_lr: float
    min_lr: float

class BaseModule(LightningModule):
    def __init__(self,
                 optim_args: AdamWCosineOptimArgs,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 header: nn.Module = None):
        super().__init__()

        self.optim_args = optim_args
        self.encoder = encoder
        self.decoder = decoder
        self.header = header

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    def forward(self, batch: dict) -> torch.Tensor:
        y_hat = self.encoder(batch)
        y_hat = self.decoder(y_hat) if self.decoder else y_hat[-1]
        y_hat = self.header(y_hat) if self.header else y_hat
        return y_hat

    def get_param_groups(self):
        """ Split params into two groups according to encoder.no_weight_decay()
        :return: List of groups with weight decay and without weight decay
        """
        def check_keywords_in_name(name_: str, skip_keywords_: set[str]):
            for keyword in skip_keywords_:
                if keyword in name_:
                    return True
            return False

        skip_keywords = self.encoder.no_weight_decay()

        has_decay_param = []
        no_decay_param = []
        has_decay_name = []
        no_decay_name = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if check_keywords_in_name(name, skip_keywords):
                no_decay_param.append(param)
                no_decay_name.append(name)
            else:
                has_decay_param.append(param)
                has_decay_name.append(name)
        print("no_decay_params:", no_decay_name)

        return [{'params': has_decay_param},
                {'params': no_decay_param, 'weight_decay': 0}]

    def configure_optimizers(self):
        if hasattr(self.encoder, "no_weight_decay"):
            params = self.get_param_groups()
        else:
            params = self.parameters()

        optimizer = optim.AdamW(params=params, lr=self.optim_args.max_lr, weight_decay=self.optim_args.weight_decay)
        lr_scheduler = CosineAnnealingWithWarmup(optimizer=optimizer, warmup_epochs=self.optim_args.warmup_epochs,
                                                 annealing_epochs=self.optim_args.annealing_epochs,
                                                 max_lr=self.optim_args.max_lr,
                                                 min_lr=self.optim_args.min_lr)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}


class ClassificationModule(BaseModule):
    def __init__(self,
                 optim_args: AdamWCosineOptimArgs,
                 encoder: nn.Module = None,
                 decoder: nn.Module = None,
                 header: nn.Module = None,
                 ignore_index: int = None):

        super().__init__(optim_args=optim_args, encoder=encoder, decoder=decoder, header=header)

        # important: ignore_index is the pixel value corresponding to the ignored class
        # can not use -1 to ignore the last class, -100 is the default value
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=ignore_index or -100)

        assert hasattr(header, "num_classes"), f"header {header.__class__} doesn't hava num_classes attribute"
        num_classes = header.num_classes
        # metrics
        global_metrics, classes_metrics, confusion_matrix = generate_classification_metric(num_classes=num_classes,
                                                                                           ignore_index=ignore_index)

        self.val_global_metric = global_metrics.clone(prefix="val_")
        self.test_global_metric = global_metrics.clone(prefix="test_")
        self.confusion_matrix = confusion_matrix

        self.val_classes_metric = classes_metrics.clone(prefix="val_")
        self.test_classes_metric = classes_metrics.clone(prefix="test_")

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)

    def training_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="train_loss", value=loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        loss = self.cross_entropy_loss(y_hat, y)
        self.log(name="val_loss", value=loss, on_epoch=True, sync_dist=True)
        self.val_global_metric.update(y_hat, y)
        self.val_classes_metric.update(y_hat, y)

    def on_validation_epoch_end(self) -> None:
        global_metric_value = self.val_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.val_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.val_global_metric.reset()
        self.val_classes_metric.reset()

    def test_step(self, batch: dict, batch_index: int):
        y_hat = self(batch)
        y = batch["y"]
        self.test_global_metric.update(y_hat, y)
        self.test_classes_metric.update(y_hat, y)

    def on_test_epoch_end(self):
        global_metric_value = self.test_global_metric.compute()
        classes_metric_value = separate_classes_metric(self.test_classes_metric.compute())

        metric_values = {**global_metric_value, **classes_metric_value}
        self.log_dict(metric_values, sync_dist=True)

        self.test_global_metric.reset()
        self.test_classes_metric.reset()

    def predict_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        y_hat = self(batch)
        y_hat = torch.argmax(y_hat, dim=1).to(torch.uint8)
        return y_hat
