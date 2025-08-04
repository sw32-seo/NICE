# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python project for practicing with the Grain data loading library using MNIST dataset. The project loads raw MNIST binary files, processes them into PyArrow format, and saves them as Parquet files for efficient data processing.

## Architecture

- **mnist_loader.py**: Main script that loads MNIST binary files (IDX format), parses headers, converts to NumPy arrays, and saves as PyArrow Parquet files
- **test.py**: Simple test script demonstrating PyArrow table creation from NumPy arrays
- **mnist/**: Directory containing raw MNIST dataset files in IDX binary format
- **output/**: Directory containing processed Parquet files (train_dataset.parquet, test_dataset.parquet)

## Key Dependencies

The project uses these main libraries:
- `grain`: Google's data loading library
- `numpy`: Array processing
- `pyarrow`: Columnar data format and Parquet file handling
- `absl`: Command-line flag handling and logging

## Running the Code

### Main Data Processing
```bash
python mnist_loader.py --data_dir=mnist --output_dir=output
```

### Test Script
```bash
pytest tests
```

## Data Flow

1. Load raw MNIST IDX files (train-images, train-labels, t10k-images, t10k-labels)
2. Parse binary headers to extract metadata
3. Convert binary data to NumPy arrays
4. Transform 2D image arrays into lists of 1D arrays for PyArrow compatibility
5. Create PyArrow tables and save as Parquet files

## Important Notes

- The project handles PyArrow compatibility by converting 2D NumPy arrays to lists of 1D arrays before creating tables
- Image data is flattened from 28x28 to 784-dimensional vectors
- Uses big-endian format ('>') for parsing IDX file headers