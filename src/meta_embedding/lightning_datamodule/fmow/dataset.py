from torch.utils.data.dataset import Dataset
import pandas as pd
import rasterio
import numpy as np
import torch
from meta_embedding.lightning_datamodule.transforms import FMOWSentinelTrainTransform, FMOWSentinelEvalTransform
import os


class FMOWSentinelDataset(Dataset):
    mean = [1370.19151926, 1184.3824625, 1120.77120066, 1136.26026392,
            1263.73947144, 1645.40315151, 1846.87040806, 1762.59530783,
            1972.62420416, 582.72633433, 14.77112979, 1732.16362238, 1247.91870117]
    std = [633.15169573, 650.2842772, 712.12507725, 965.23119807,
           948.9819932, 1108.06650639, 1258.36394548, 1233.1492281,
           1364.38688993, 472.37967789, 14.3114637, 1310.36996126, 1087.6020813]
    categories = ["airport", "airport_hangar", "airport_terminal", "amusement_park",
     "aquaculture", "archaeological_site", "barn", "border_checkpoint",
     "burial_site", "car_dealership", "construction_site", "crop_field",
     "dam", "debris_or_rubble", "educational_institution", "electric_substation",
     "factory_or_powerplant", "fire_station", "flooded_road", "fountain",
     "gas_station", "golf_course", "ground_transportation_station", "helipad",
     "hospital", "impoverished_settlement", "interchange", "lake_or_pond",
     "lighthouse", "military_facility", "multi-unit_residential",
     "nuclear_powerplant", "office_building", "oil_or_gas_facility", "park",
     "parking_lot_or_garage", "place_of_worship", "police_station", "port",
     "prison", "race_track", "railway_bridge", "recreational_facility",
     "road_bridge", "runway", "shipyard", "shopping_mall",
     "single-unit_residential", "smokestack", "solar_farm", "space_facility",
     "stadium", "storage_tank", "surface_mine", "swimming_pool", "toll_booth",
     "tower", "tunnel_opening", "waste_disposal", "water_treatment_facility",
     "wind_farm", "zoo"]

    def __init__(self,
                 sentinel_root_path: str,
                 csv_path: str,
                 choice_percent: float,
                 input_size : int,
                 is_train: bool,
                 masked_bands: list[int] | None,
                 dropped_bands: list[int] | None):
        """
        Create FMoW sentinel dataset using SatMae Setting
        """
        self.sentinel_root_path = sentinel_root_path
        self.df = pd.read_csv(csv_path).sort_values(["category", "location_id", "timestamp"])
        self.indices = self.random_choose_indices(choice_percent)

        if is_train:
            self.image_transform = FMOWSentinelTrainTransform(self.mean, self.std, input_size)
        else:
            self.image_transform = FMOWSentinelEvalTransform(self.mean, self.std, input_size)

        self.masked_bands = masked_bands
        self.dropped_bands = dropped_bands

    def __len__(self):
        return len(self.df)

    def random_choose_indices(self, choice_percent: float) -> np.ndarray:
        indexes = np.array([])  # Empty numpy array to store the indexes

        for value in self.df['category'].unique():
            category_rows = self.df[self.df['category'] == value]  # Get rows belonging to a specific category
            rows_to_select = int(len(category_rows) * 0.1)  # Calculate 10% of the rows for selection
            selected_indexes = np.random.choice(category_rows.index, size=rows_to_select,
                                                replace=False)  # Randomly select indexes
            indexes = np.concatenate((indexes, selected_indexes))  # Append selected indexes to the numpy array

    def get_image_path(self, index: int) -> str:
        row = self.df.iloc[index]
        category = row["category"]
        location_id = row["location_id"]
        image_id = row["image_id"]

        result = os.path.join(self.sentinel_root_path,
                              category,
                              f"{category}_{location_id}",
                              f"{category}_{location_id}_{image_id}.tif")
        return result

    def get_x(self, image_path: str) -> torch.Tensor:
        with rasterio.open(image_path) as data:
            image = data.read()

        if self.masked_bands is not None:
            image[self.masked_bands, :, :] = np.array(self.mean)[self.masked_bands]
        result =  self.image_transform(image)

        if self.dropped_bands is not None:
            keep_indexes = [i for i in range(result.shape[0]) if i not in self.dropped_bands]
            result = result[keep_indexes, :, :]
        return result


    def __getitem__(self, index):
        x = self.get_x(self.get_image_path(index))
        y = self.categories.index(self.df.iloc[index]["category"])

        result = {"x": x, "y": y,}
        return result

def main():
    dataset = FMOWSentinelDataset(sentinel_root_path="/mnt/disk/xials/dataset/fmow/fmow-sentinel/val",
                                  csv_path="/mnt/disk/xials/dataset/fmow/val.csv",
                                  input_size=96,
                                  is_train=False,
                                  masked_bands=None,
                                  dropped_bands=[0,9,10])
    sample0 = dataset[0]
    x = sample0["x"]
    y = sample0["y"]
    print(x.shape, x.min(), x.max(), x.mean(), x.std())

if __name__ == "__main__":
    main()