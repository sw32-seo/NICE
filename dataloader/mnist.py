import grain
import numpy as np


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


def get_mnist_dataloader(batch_size=512, seed=42):
    train_ds = grain.experimental.ParquetIterDataset("output/train_dataset.parquet").map(Dequantize(seed)).batch(
        batch_size, drop_remainder=True)
    valid_ds = grain.experimental.ParquetIterDataset("output/valid_dataset.parquet").map(Dequantize(seed)).batch(
        batch_size, drop_remainder=False)
    test_ds = grain.experimental.ParquetIterDataset("output/test_dataset.parquet").map(Dequantize(seed)).batch(
        batch_size, drop_remainder=False)
    return train_ds, valid_ds, test_ds
