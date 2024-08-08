"""
Contains functions for testing a PyTorch model.
"""
from typing import Tuple, List

import torch
import numpy as np
import pickle


from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              use_mixed_precision: bool = False) -> Tuple[float, float, List[int], List[int]]:
    """Tests a PyTorch model.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a testing dataset.

    Args:
        model: A PyTorch model to be tested.
        dataloader: A DataLoader instance for the model to be tested on.
        loss_fn: A PyTorch loss function to calculate loss on the test data.
        device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
        A tuple of testing loss and testing accuracy metrics.
        In the form (test_loss, test_accuracy). For example:

        (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Lists to store true and predicted labels
    y_labels = []
    y_preds = []

    # Turn on inference context manager
    with torch.inference_mode():  # similar to torch.no_grad()
        with torch.cuda.amp.autocast(enabled=use_mixed_precision):
            # Loop through DataLoader batches
            for batch, (X, y) in enumerate(dataloader):
                # Send data to target device
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred_logits = model(X)

                # 2. Calculate and accumulate loss
                loss = loss_fn(test_pred_logits, y)
                test_loss += loss.detach().cpu().item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

                # Collect labels and predictions
                y_labels.extend(y.detach().cpu().numpy())
                y_preds.extend(test_pred_labels.detach().cpu().numpy())

    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, y_labels, y_preds
