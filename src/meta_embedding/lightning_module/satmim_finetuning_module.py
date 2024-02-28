from .satmim_pretrain_module import SatMIMPreTrainingModule
from .classification_module import ClassificationModule
from .base_module import AdamWCosineOptimArgs

from torch import nn

__all__ = ["SatMIMFineTuningModule"]


class SatMIMFineTuningModule(ClassificationModule):
    def __init__(self, pretrain_ckpt_path: str, is_classify: bool, optim_args: AdamWCosineOptimArgs,
                 encoder: nn.Module, decoder: nn.Module = None, header: nn.Module = None, ):
        super().__init__(optim_args=optim_args, encoder=encoder, decoder=decoder, header=header)
        self.is_classify = is_classify
        # load encoder
        pretrain_module = SatMIMPreTrainingModule.load_from_checkpoint(pretrain_ckpt_path)
        msg = self.encoder.load_state_dict(pretrain_module.encoder.state_dict(), strict=True)
        print(msg)

    def forward(self, batch: dict):
        batch["mask"] = None
        y_hat = self.encoder(batch, is_pretrain=False, is_classify=self.is_classify)

        y_hat = self.decoder(y_hat) if self.decoder else y_hat[-1]
        y_hat = self.header(y_hat) if self.header else y_hat
        return y_hat




