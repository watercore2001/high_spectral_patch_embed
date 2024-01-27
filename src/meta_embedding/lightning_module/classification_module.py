import torch
import torch.nn as nn
from .base_module import AdamWCosineOptimArgs, BaseModule
from .metrics import generate_classification_metric, separate_classes_metric
from typing import Any

__all__ = ["ClassificationModule"]


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

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        y_hat = self(batch)
        y_hat = torch.argmax(y_hat, dim=1).to(torch.uint8)
        return y_hat
