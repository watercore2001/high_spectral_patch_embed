import os

import torch
import urllib3
from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.cli import LightningCLI, LightningArgumentParser
from pytorch_lightning.loggers import WandbLogger

urllib3.disable_warnings()


class LoggerLightningCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--used_ckpt_for_test", choices=["best", "last", "no"])
        parser.add_argument("--used_ckpt_for_predict", choices=["best", "last", "no"])

        parser.add_class_arguments(WandbLogger, nested_key="wandb_logger")
        parser.link_arguments(source="trainer.default_root_dir", target="wandb_logger.save_dir")
        parser.link_arguments(source="wandb_logger.name", target="wandb_logger.version")

    def instantiate_trainer(self, **kwargs) -> Trainer:
        subcommand = self.config_init["subcommand"]
        config = self.config_init[subcommand]

        wandb_logger = config["wandb_logger"]
        trainer_logger = config["trainer"]["logger"]

        # save all parameter in workspace
        os.makedirs(os.path.join(wandb_logger.save_dir, wandb_logger.name, wandb_logger.version), exist_ok=True)
        self.save_config_kwargs.update({"config_filename": f"{wandb_logger.name}/{wandb_logger.version}/config.yaml"})

        # trainer_logger's default value is True
        if wandb_logger is not None and trainer_logger is True:
            config["trainer"]["logger"] = wandb_logger

        return super().instantiate_trainer()

    def after_fit(self):
        subcommand = self.config_init["subcommand"]
        used_ckpt_for_test = self.config_init[subcommand]["used_ckpt_for_test"]
        used_ckpt_for_predict = self.config_init[subcommand]["used_ckpt_for_predict"]

        if used_ckpt_for_test != "no":
            self.trainer.test(model=self.model, datamodule=self.datamodule, ckpt_path=used_ckpt_for_test)
        if used_ckpt_for_predict != "no":
            self.trainer.predict(model=self.model, datamodule=self.datamodule, ckpt_path=used_ckpt_for_predict)


def main():
    # lower precision for higher speed
    torch.set_float32_matmul_precision("medium")

    LoggerLightningCLI(
        LightningModule,
        LightningDataModule,
        save_config_kwargs={"overwrite": True},
        subclass_mode_model=True,
        subclass_mode_data=True
    )


if __name__ == '__main__':
    main()
