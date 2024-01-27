from pytorch_lightning import LightningDataModule
from typing import Literal
import os
from torch.utils.data import DataLoader
import dataclasses
from meta_embedding.lightning_datamodule.fmow.dataset import FMOWSentinelDataset

@dataclasses.dataclass
class DataloaderArgs:
    batch_size: int
    # must set to 0 in debug mode
    num_workers: int
    pin_memory: bool
    shuffle: bool
    # if the last batch size == 1, UperNet will crash because batch normalization
    drop_last: bool = True

class FMOWSentinelDataModule(LightningDataModule):
    def __init__(self,
                 root_path: str,
                 input_size: int,
                 masked_bands: list[int] | None,
                 dropped_bands: list[int] | None,
                 dataloader_args: DataloaderArgs):
        super().__init__()
        self.root_path = root_path
        self.input_size = input_size
        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands
        self.dataloader_args = dataloader_args


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def make_dataset(self, stage_name: Literal["train", "val", "test_gt"]):
        stage_sentinel_folders = os.path.join(self.root_path, "fmow-sentinel", stage_name)
        csv_path = os.path.join(self.root_path, f"{stage_name}.csv")
        return FMOWSentinelDataset(sentinel_root_path=stage_sentinel_folders,
                                   csv_path = csv_path,
                                   input_size = self.input_size,
                                   is_train = (stage_name=="train"),
                                   masked_bands = self.masked_bands,
                                   dropped_bands = self.dropped_bands)

    def setup(self, stage: [str] = None):
        if stage == "fit":
            self.train_dataset = self.make_dataset("train")
            self.val_dataset = self.make_dataset("val")
        if stage == "test":
            self.test_dataset = self.make_dataset("test_gt")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, **dataclasses.asdict(self.dataloader_args))

    def val_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.val_dataset, **dataclasses.asdict(dataloader_args))

    def predict_dataloader(self):
        dataloader_args = dataclasses.replace(self.dataloader_args, shuffle=False)
        return DataLoader(self.test_dataset, **dataclasses.asdict(dataloader_args))

def main():
    module = FMOWSentinelDataModule(root_path="/mnt/disk/xials/dataset/fmow",
                                    input_size=96,
                                    masked_bands=None,
                                    dropped_bands=[0,9,10],
                                    dataloader_args = DataloaderArgs(batch_size=4, num_workers=0, pin_memory=True, shuffle=False))
    sample0 = dataset[0]
    x = sample0["x"]
    y = sample0["y"]
    print(x.shape, x.min(), x.max(), x.mean(), x.std())