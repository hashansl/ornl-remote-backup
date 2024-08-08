"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os
import torch
import numpy as np
import data_loader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

# NUM_WORKERS =0

def create_dataloaders(
    annotation_file_path: str, 
    root_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """

  #---
  data_set = data_loader.data_loader_persistence_img(annotation_file_path=annotation_file_path, root_dir=root_dir, transform=transform)

  #---
  # Set the random seed
  torch.manual_seed(42)
  np.random.seed(42)

  #---
  total_size = len(data_set)
  train_size = int(0.75 * total_size)
  val_size = int(0.15 * total_size)
  test_size = total_size - train_size - val_size

  #---
  train_set, val_set, test_set = torch.utils.data.random_split(data_set, [train_size, val_size, test_size])



  #---
  # train_data = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
  # test_data = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False)



  # # Use ImageFolder to create dataset(s)
  # train_data = datasets.ImageFolder(train_dir, transform=transform)
  # test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = data_set.get_class_names()

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )

  validation_dataloader = DataLoader(
      val_set,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle ???
      num_workers=num_workers,
      pin_memory=True,
  )

  test_dataloader = DataLoader(
      test_set,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, validation_dataloader, test_dataloader, class_names