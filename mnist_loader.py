import grain
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt

train_ds = grain.experimental.ParquetIterDataset("output/train_dataset.parquet")
test_ds = grain.experimental.ParquetIterDataset("output/test_dataset.parquet")


class Dequantize(grain.transforms.Map):

    def __init__(self, seed=42):
        super().__init__()
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def map(self, element):
        dequantized_image = element['image'] + self.rng.uniform(0, 1. / 256., size=element['image'].shape)
        zero_one_min_max_scaled_image = (dequantized_image - dequantized_image.min()) / (dequantized_image.max() -
                                                                                         dequantized_image.min())
        return {'image': zero_one_min_max_scaled_image, 'label': element['label']}


train_ds = train_ds.map(Dequantize(42))
test_ds = test_ds.map(Dequantize(42))
