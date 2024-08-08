"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import numpy as np
import pickle
from sklearn.metrics import f1_score

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,
               use_mixed_precision: bool = False) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train() 

# When you call model.train(), you're essentially telling the model that you're starting 
# the training phase, and it should activate certain behaviors that are typically used during training, 
# such as enabling dropout layers for stochastic regularization and batch normalization layers to use batch 
# statistics for normalization.
# This is important because some operations, like dropout, behave differently during training and evaluation. 
# During training, dropout randomly zeroes some of the elements of the input tensor with a certain probability
# to prevent overfitting. However, during evaluation, dropout is usually turned off so that the model 
# produces consistent predictions.

  # Setup train loss and train accuracy values
  train_loss, train_acc, train_f1 = 0, 0, 0

  # Initialize GradScaler for mixed precision
  scaler = torch.cuda.amp.GradScaler(enabled=use_mixed_precision)

  all_labels = []
  all_preds = []

  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
    with torch.cuda.amp.autocast(enabled=use_mixed_precision):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # # Debugging: Print X and y values
        # print(f"Batch {batch} - y: {y}")
        
        # # Check for NaNs in X and y
        # if torch.isnan(X).any() or torch.isnan(y).any():
        #     print(f"NaNs detected in input data at batch {batch}")

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)

        # Debugging: Print the loss value
        # print(f"Batch {batch} - Loss: {loss.item()}")
        
        # if torch.isnan(loss):
        #     print(f"NaNs detected in loss at batch {batch}")
        #     print(f"y: {y}")
        #     print(f"y_pred: {y_pred}")
        #     print(f"X: {X}")
            


        train_loss += loss.detach().cpu().item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        scaler.scale(loss).backward()
        
        # 5. Optimizer step
        scaler.step(optimizer)
        scaler.update()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Calculate and accumulate f1 score
        all_labels.extend(y.detach().cpu().numpy())
        all_preds.extend(y_pred_class.detach().cpu().numpy())

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  #f1 score
  train_f1 = f1_score(all_labels, all_preds, average='weighted')

  return train_loss, train_acc, train_f1

def validation_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
              use_mixed_precision: bool = False) -> Tuple[float, float]:
  """validations a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a validationing dataset.

  Args:
    model: A PyTorch model to be validationed.
    dataloader: A DataLoader instance for the model to be validationed on.
    loss_fn: A PyTorch loss function to calculate loss on the validation data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of validationing loss and validationing accuracy metrics.
    In the form (validation_loss, validation_accuracy). For example:

    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 

  # Setup validation loss and validation accuracy values
  validation_loss, validation_acc, validation_f1 = 0, 0, 0

  all_labels = []
  all_preds = []
  

  # Turn on inference context manager
  with torch.inference_mode(): # similalr to torch.no_grad()
    with torch.cuda.amp.autocast(enabled=use_mixed_precision):
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            validation_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(validation_pred_logits, y)
            validation_loss += loss.detach().cpu().item()

            # Calculate and accumulate accuracy
            validation_pred_labels = validation_pred_logits.argmax(dim=1)
            validation_acc += ((validation_pred_labels == y).sum().item()/len(validation_pred_labels))

            # Calculate and accumulate f1 score
            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(validation_pred_labels.detach().cpu().numpy())

  # Adjust metrics to get average loss and accuracy per batch 
  validation_loss = validation_loss / len(dataloader)
  validation_acc = validation_acc / len(dataloader)

  validation_f1 = f1_score(all_labels, all_preds, average='weighted')

  
  return validation_loss, validation_acc, validation_f1

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          validation_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device,
          use_mixed_precision:False,
          save_name:str,
          save_path) -> Dict[str, List]:
  """Trains and validations a PyTorch model.

  Passes a target PyTorch models through train_step() and validation_step()
  functions for a number of epochs, training and validationing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and validationed.
    train_dataloader: A DataLoader instance for the model to be trained on.
    validation_dataloader: A DataLoader instance for the model to be validationed on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and validationing loss as well as training and
    validationing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  validation_loss: [...],
                  validation_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  validation_loss: [1.2641, 1.5706],
                  validation_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "validation_loss": [],
      "validation_acc": []
  }

  # Initialize patience counter
  patience_ctr = 0

  # Initialize best loss to infinity
  best_loss = np.inf

  # Loop through training and validationing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc, train_f1 = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device
                                          ,use_mixed_precision=use_mixed_precision)
      validation_loss, validation_acc, validation_f1 = validation_step(model=model,
          dataloader=validation_dataloader,
          loss_fn=loss_fn,
          device=device,
          use_mixed_precision=use_mixed_precision)

      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"train_f1: {train_f1:.4f} |"
          f"validation_loss: {validation_loss:.4f} | "
          f"validation_acc: {validation_acc:.4f} | "
          f"validation_f1: {validation_f1:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["validation_loss"].append(validation_loss)
      results["validation_acc"].append(validation_acc)

      # Check for the best validation loss
      if epoch == 1:
          best_loss = validation_loss

          # saving the model(only the weights) for the first time
          # This can be deleted if you don't want to save the model for the first time
          torch.save({
                 "epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "validation_loss": best_loss,
              },(save_path+save_name))
          print("Model saved!")
      else:
          if validation_loss < best_loss:
              best_loss = validation_loss
              torch.save({
                 "epoch": epoch,
                  "model_state_dict": model.state_dict(),
                  "optimizer_state_dict": optimizer.state_dict(),
                  "validation_loss": best_loss,
              },(save_path+save_name))
              print("Model saved!")
              patience_ctr = 0  # Reset patience counter if the model improves
          # else:
          #     patience_ctr += 1
          #     if patience_ctr >= 10:
          #         print(f"Stopping criterion reached: validation loss has not improved in {patience_ctr} epochs")
          #         print(f"saving to {save_path+save_name}", flush=True)

          #         # loading weights of best model
          #         checkpoint = torch.load(save_path+save_name)
          #         model.load_state_dict(checkpoint["model_state_dict"])

          #         with open(save_path+str(best_loss), "wb") as f_out:
          #           pickle.dump(results, f_out, pickle.HIGHEST_PROTOCOL)
          #         break
  if epoch+1 == epochs:
    print("Model training hit max epochs, not converged")

    #loading weights of best model
    checkpoint = torch.load(save_path+save_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"saving to {save_path+save_name}", flush=True)
    with open(save_path+str(best_loss), "wb") as f_out:
      pickle.dump(results, f_out, pickle.HIGHEST_PROTOCOL)

  # Return the filled results at the end of the epochs
  return results