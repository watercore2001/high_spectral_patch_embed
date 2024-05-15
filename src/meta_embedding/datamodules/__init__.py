import dataclasses


@dataclasses.dataclass
class DataloaderArgs:
    batch_size: int
    # must set to 0 in debug mode
    num_workers: int
    pin_memory: bool
    shuffle: bool
