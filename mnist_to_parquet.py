from pprint import pprint
import grain
from absl import app, flags, logging
import numpy as np
import struct
import pyarrow as pa
import pyarrow.parquet as pq

FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', 'mnist', 'Directory to download data to.')
flags.DEFINE_string('output_dir', 'output', 'Directory to save output to.')


def parse_header(header_bytes, image_or_label):
    if image_or_label == 'image':
        magic_number, num_images, num_rows, num_cols = struct.unpack('>IIII', header_bytes)
    elif image_or_label == 'label':
        magic_number, num_labels = struct.unpack('>II', header_bytes)
    else:
        raise ValueError(f"Invalid image_or_label: {image_or_label}")
    print("-" * 100)
    print(f"Magic Number: {magic_number}")
    if image_or_label == 'image':
        print(f"Number of Images: {num_images}")
        print(f"Number of Rows: {num_rows}")
        print(f"Number of Columns: {num_cols}")
    elif image_or_label == 'label':
        print(f"Number of Labels: {num_labels}")
    print("-" * 100)


def load_mnist(data_dir):
    image_size = 28

    with open(data_dir + '/train-images.idx3-ubyte', 'rb') as f:
        train_images_header = f.read(16)
        print("train_images_header:")
        parse_header(train_images_header, 'image')
        train_images = f.read()
    with open(data_dir + '/train-labels.idx1-ubyte', 'rb') as f:
        train_labels_header = f.read(8)
        print("train_labels_header:")
        parse_header(train_labels_header, 'label')
        train_labels = f.read()
    with open(data_dir + '/t10k-images.idx3-ubyte', 'rb') as f:
        test_images_header = f.read(16)
        print("test_images_header:")
        parse_header(test_images_header, 'image')
        test_images = f.read()
    with open(data_dir + '/t10k-labels.idx1-ubyte', 'rb') as f:
        test_labels_header = f.read(8)
        print("test_labels_header:")
        parse_header(test_labels_header, 'label')
        test_labels = f.read()

    train_images = np.frombuffer(train_images, dtype=np.uint8).reshape(-1, image_size * image_size)
    train_labels = np.frombuffer(train_labels, dtype=np.uint8).reshape(-1)
    test_images = np.frombuffer(test_images, dtype=np.uint8).reshape(-1, image_size * image_size)
    test_labels = np.frombuffer(test_labels, dtype=np.uint8).reshape(-1)

    return train_images, train_labels, test_images, test_labels


def main(_):
    logging.info(f'data_dir: {FLAGS.data_dir}')
    logging.info(f'output_dir: {FLAGS.output_dir}')

    train_images, train_labels, test_images, test_labels = load_mnist(FLAGS.data_dir)

    pprint(train_images.shape)
    pprint(train_labels.shape)
    pprint(test_images.shape)
    pprint(test_labels.shape)

    # Randomly select validation set from train_images and train_labels to have same size as test_images and test_labels
    val_size = test_images.shape[0]
    val_indices = np.random.choice(train_images.shape[0], val_size, replace=False)
    val_images, val_labels = train_images[val_indices], train_labels[val_indices]
    train_indices = np.setdiff1d(np.arange(train_images.shape[0]), val_indices)
    train_images, train_labels = train_images[train_indices], train_labels[train_indices]
    pprint(train_images.shape)
    pprint(train_labels.shape)
    pprint(val_images.shape)
    pprint(val_labels.shape)

    # Make a pyarrow table of train_images and train_labels
    train_table = pa.Table.from_pydict({'image': [row for row in train_images], 'label': train_labels})
    valid_table = pa.Table.from_pydict({'image': [row for row in val_images], 'label': val_labels})
    test_table = pa.Table.from_pydict({'image': [row for row in test_images], 'label': test_labels})

    with pq.ParquetWriter(FLAGS.output_dir + '/train_dataset.parquet', train_table.schema) as writer:
        writer.write_table(train_table)
    with pq.ParquetWriter(FLAGS.output_dir + '/valid_dataset.parquet', valid_table.schema) as writer:
        writer.write_table(valid_table)
    with pq.ParquetWriter(FLAGS.output_dir + '/test_dataset.parquet', test_table.schema) as writer:
        writer.write_table(test_table)

    # load the parquet files
    train_ds = pq.read_table(FLAGS.output_dir + '/train_dataset.parquet')
    valid_ds = pq.read_table(FLAGS.output_dir + '/valid_dataset.parquet')
    test_ds = pq.read_table(FLAGS.output_dir + '/test_dataset.parquet')
    pprint(train_ds)
    pprint(valid_ds)
    pprint(test_ds)


if __name__ == '__main__':
    app.run(main)
