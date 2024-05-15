from pytorch_lightning import LightningDataModule
from meta_embedding.datamodules.seg_munich.dataset import SegDataset
from meta_embedding.datamodules import DataloaderArgs
import dataclasses
from torch.utils.data import DataLoader


class SegMunichDataModule(LightningDataModule):
    def __init__(self,
                 root_path: str,
                 dataloader_args: DataloaderArgs):
        super().__init__()
        self.root_path = root_path
        self.dataloader_args = dataloader_args

        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = SegDataset(self.root_path, txt_name="train.txt", training=True)
            self.val_dataset = SegDataset(self.root_path, txt_name="val.txt", training=False)
        if stage == "test":
            self.val_dataset = SegDataset(self.root_path, txt_name="val.txt", training=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.val_dataset, **dataclasses.asdict(dataloader_args))

    def test_dataloader(self):
        return self.val_dataloader()


if __name__ == '__main__':
    module = SegMunichDataModule(root_path="/mnt/data/dataset/TUM_128",
                                 dataloader_args=DataloaderArgs(batch_size=4, num_workers=0, pin_memory=True, shuffle=False))
    module.setup(stage="fit")
