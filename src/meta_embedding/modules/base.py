import dataclasses

import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch import optim

from .lr_scheduler import CosineAnnealingWithWarmup

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
                 decoder: nn.Module = None):
        super().__init__()

        self.optim_args = optim_args
        self.encoder = encoder
        self.decoder = decoder

        # must save all hyperparameters for checkpoint
        self.save_hyperparameters(logger=False)



    @torch.jit.ignore
    def no_weight_decay(self):
        result = set()
        for module in [self.encoder, self.decoder]:
            if module is not None and hasattr(module, "no_weight_decay"):
                result = result | module.no_weight_decay()
        return result

    def get_param_groups(self):
        """ Split params into two groups according to encoder.no_weight_decay()
        :return: List of groups with weight decay and without weight decay
        """

        def check_keywords_in_name(name_: str, skip_keywords_: set[str]):
            for keyword in skip_keywords_:
                if keyword in name_:
                    return True
            return False

        skip_keywords = self.no_weight_decay()

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
