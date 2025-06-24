'''
Workflow:
1. Load Logger (logs everything to console and logfile)
2. Set random seeds for reproducability
3. Set Hyperparameters
4. Load training and validation data
5. Initialize model, loss function, optimizer, metrics
6. Train model
7. Save model
'''

# built-in modules
import logging
import random
from datetime import datetime
import traceback
import time
import os

# 3rd party modules
import numpy as np
import torch
# import albumentations as A
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchmetrics.classification import MulticlassJaccardIndex
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
from pathlib import Path

# own modules
from src.ml.ml_datasets import OCTDataset

# Globals
#augmentations = A.Compose([
#     A.SomeOf([
#         A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.3), contrast_limit=(-0.3, 0.3), p=0.4),
#         A.GaussianBlur(blur_limit=(3, 3), p=0.2),
#         A.HorizontalFlip(p=0.4),
#         A.GaussNoise(std_range=(0.05, 0.2), mean_range=(0.01, 0.05), p=0.2),
#     ], n=3, p=1)
# ])
now = datetime.now().strftime('%Y_%m_%d-%H.%M')
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = Path(current_dir).parent.parent
rand_seed = 37

# Data directories
# train_imgs = os.path.join(root_dir, 'data', 'biofilm_on_membrane_with_spacer', 'labeled', 'train_images')
# train_masks = os.path.join(root_dir, 'data', 'biofilm_on_membrane_with_spacer', 'labeled', 'train_masks')
train_imgs = input('Enter path to training images: ').strip()
train_masks = input('Enter path to training masks: ').strip()

# val_imgs = os.path.join(root_dir, 'data', 'biofilm_on_membrane_with_spacer', 'labeled', 'val_images')
# val_masks = os.path.join(root_dir, 'data', 'biofilm_on_membrane_with_spacer', 'labeled', 'val_masks')
val_imgs = input('Enter path to validation images: ').strip()
val_masks = input('Enter path to validation masks: ').strip()

# 1. Set up logging
logfile = os.path.join(current_dir, 'ml_logs', f'{now}_training_log.txt')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler(logfile, mode='w')]
    )
logger = logging.getLogger()

# Program entry point --------------------------------------------------------------------------------------------------
def main():
    logger.info(f'Logging to file: {logfile}')
    logger.info('Initializing training sequence for OCT semantic segmentation model...')

    # 2. Set random seeds for reproducability
    set_random_seeds(rand_seed)
    logger.info(f'Random seeds set for reproducability: {rand_seed}')

    # 3. Set parameters
    # 3.1 Set model architecture
    encoder = 'efficientnet-b2' # available: b0, b1, b2, b3, b4, b5, b6, b7 (model size increases with number)
    weights = 'imagenet'
    colorchannels = 1
    activation = None
    depth = 5

    # 3.2 Set Hyperparameters
    batch_size = 1
    virtual_batch_size = 11
    class_labels = ['background', 'biofilm', 'optical window', 'membrane', 'spacer']
    n_classes = len(class_labels)
    cls_dist_dict = {
        'background': 0.6,
        'biofilm': 0.15,
        'membrane': 0.05,
        'optical window': 0.05,
        'spacer': 0.15
    }
    epochs = 200 
    patience = 300
    learnrate = 0.001
    parallel_processes = 0
    class_distribution = torch.tensor([cls_dist_dict[cls] for cls in class_labels])
    inverse_class_distribution = 1 / class_distribution # inverse class distribution for class weights
    norm_weights = inverse_class_distribution / inverse_class_distribution.sum() # normalize class weights to sum to 1
    class_weights = norm_weights * n_classes # Gewichtung beim Training --> inverse class distribution

    logger.info(f'Hyperparameters initialized: {locals()}')


    # 4. Load training and validation data
    # Training dataset --> model learns from this data
    train_dataset = OCTDataset(train_imgs, train_masks)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=parallel_processes)
    logger.info(f'Training dataset will load from: <<{train_imgs}>> and <<{train_masks}>>')

    # Validation dataset --> "testing" model accuracy after each epoch, model doesnt learn from this data
    val_dataset = OCTDataset(val_imgs, val_masks)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=parallel_processes)
    logger.info(f'Validation dataset will load from: <<{val_imgs}>> and <<{val_masks}>>.')

    # 5. Initialize model, loss function, optimizer, metrics, device
    model = build_model(encoder, weights, colorchannels, n_classes, activation, depth)
    model.segmentation_head[0].bias.data = torch.log(class_distribution) # set bias -> good starting point for training
    optimizer = optim.Adam(model.parameters(), lr=learnrate)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    iou_metric = MulticlassJaccardIndex(num_classes=n_classes, average=None)
    logger.info('Model, loss function, optimizer and metrics initialized.')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # train on GPU or CPU
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    iou_metric = iou_metric.to(device)
    logger.info(f'Training on device: {device}')

    # 6. Train model
    t0 = time.perf_counter()
    logger.info('Initialization completed. Starting training now...')

    model = train_model(
        model=model,
        training_dataloader=train_dataloader,
        validation_dataloader=val_dataloader,
        n_epochs=epochs,
        optimizer=optimizer,
        metric=iou_metric,
        loss_function=loss_fn,
        device=device,
        patience=patience,
        batch_size=virtual_batch_size,
        num_classes=n_classes,
        class_labels=class_labels
    )
    t1 = time.perf_counter()
    logger.info(f'Model training completed in {t1-t0:.1f} seconds.')
    
    # 7. Save model
    model_dir = os.path.join(current_dir, 'ml_models')
    os.makedirs(model_dir, exist_ok=True)
    modelpath = os.path.join(model_dir, f'{now}_model.pth')
    torch.save(model.state_dict(), modelpath)
    logger.info(f'Model saved as {modelpath}.')
    logger.info(f'Logfile saved as {logfile}.')
###

def build_model(encoder_name, encoder_weights, in_channels, classes, activation, encoder_depth):
    """
    Returns a Unet model.
    """
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        encoder_depth=encoder_depth,
    )

    return model
#

def calc_class_iou(prediction, mask, num_classes):
    ious = []
    prediction = prediction.view(-1) # flatten prediction tensor to 1D
    mask = mask.view(-1) # flatten mask tensor to 1D
    for cls in range(num_classes):
        pred_inds = (prediction == cls)
        target_inds = (mask == cls)
        intersection = (pred_inds & target_inds).float().sum().item() # pixels correctly classified as cls
        union = (pred_inds | target_inds).float().sum().item() # pixels classified as cls or ground truth cls
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return ious # List: [IoU_cls_0, IoU_cls_1, ...]
#

def calc_class_F1(prediction, mask, cls):
    ...
#

def train_one_epoch(
        model,
        dataloader,
        optimizer,
        loss_fn,
        virtual_batch_size,
        device,
        metric
    ):

    model.train() # put model in training mode

    running_loss = 0.0
    batch_count = 0

    optimizer.zero_grad() # reset gradients before starting backward pass

    for i, (image, mask) in enumerate(dataloader):
        image = image.to(device)
        mask = mask.to(device)

        prediction = model(image) # outputs = predicted class probabilities (apply softmax --> confidence map)
        loss = loss_fn(prediction, mask) # calculate loss
        loss.backward() # calculate gradient of current batch, gradients accumulate until cleared by optim or zero_grad

        if (i + 1) % virtual_batch_size == 0 or (i + 1) == len(dataloader): # accumulate gradient of virtual_batch_size
            optimizer.step() # update model params
            optimizer.zero_grad() # reset gradients to 0

        # calculate class IoU
        pred_labels = prediction.argmax(1) # argmax(1) converts model output into class predictions
        metric.update(pred_labels, mask)

        running_loss += loss.item()

        batch_count += 1

    batch_loss = (running_loss / batch_count)
    batch_iou_cls = metric.compute() # calculate class IoU

    return batch_loss, batch_iou_cls
#

def evaluate(
        model,
        dataloader,
        loss_fn,
        device,
        metric
    ):

    model.eval() # put model in evaluation mode
    running_loss = 0.0
    batch_count = 0

    with torch.no_grad(): # disable gradient computation, not needed for evaluation
        for image, mask in dataloader:
            image = image.to(device)
            mask = mask.to(device)
            
            prediction = model(image)
            loss = loss_fn(prediction, mask)

            # iou = metric(prediction.argmax(1), mask)
            # total_iou += iou.item()

            # calculate class IoU
            pred_labels = prediction.argmax(1)
            metric.update(pred_labels, mask)

            running_loss += loss.item()

            batch_count += 1

    batch_loss = (running_loss / batch_count)
    batch_iou_cls = metric.compute() # calculate class IoU

    return batch_loss, batch_iou_cls
#

def train_model(
        model,
        training_dataloader,
        validation_dataloader,
        n_epochs,
        optimizer,
        metric,
        loss_function,
        device,
        patience,
        batch_size,
        num_classes,
        class_labels=None
    ):

    best_loss = float('inf') # initialize loss to +inf, so the first real loss is lower
    patience_counter = 0
    best_model_state = None

    train_losses = []
    train_ious_cls = [[] for _ in range(num_classes)]

    val_losses = []
    val_ious_cls = [[] for _ in range(num_classes)]

    plt.ion()
    fig, (ax_loss, ax_ious) = plt.subplots(1, 2, figsize=(12, 5))

    for epoch in range(n_epochs):
        t0_epoch = time.perf_counter()

        metric.reset()
        train_loss, train_iou_cls = train_one_epoch(
            model,
            training_dataloader,
            optimizer,
            loss_function,
            batch_size,
            device,
            metric
        )
        logger.info(f'Epoch {epoch+1}/{n_epochs}: train_loss= {train_loss:.4f}')

        metric.reset()
        val_loss, val_iou_cls = evaluate(model, validation_dataloader, loss_function, device, metric)
        logger.info(f'Epoch {epoch+1}/{n_epochs}: val_loss= {val_loss:.4f}')

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        for c in range(num_classes):
            train_ious_cls[c].append(train_iou_cls[c].item())
            val_ious_cls[c].append(val_iou_cls[c].item())

        plot_metrics(
            ax_loss,
            ax_ious,
            train_losses,
            val_losses,
            train_ious_cls,
            val_ious_cls,
            num_classes,
            class_labels
        )

        # Check for early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_state = model.state_dict()
            patience_counter = 0
            # Save model checkpoint
            model_dir = os.path.join(current_dir, 'ml_models')
            os.makedirs(model_dir, exist_ok=True)
            modelpath = os.path.join(model_dir, f'{now}_checkpoint_model.pth')
            torch.save(best_model_state, modelpath)
            logger.info(f'Best model checkpoint saved as {modelpath}.')
        else:
            patience_counter +=1
            if patience_counter >= patience:
                logger.info('Training stopped due to early stopping.')
                break

        t1_epoch = time.perf_counter()
        logger.info(f'Epoch {epoch+1} completed in {t1_epoch-t0_epoch:.1f} seconds.')
        #

    plt.ioff()
    plt.show() # show final plot

    log_dir = os.path.join(current_dir, 'ml_logs')
    os.makedirs(log_dir, exist_ok=True)
    figpath = os.path.join(log_dir, f'{now}_training_metrics.png')
    fig.savefig(figpath)
    logger.info(f'Training metrics plot saved as {figpath}.')

    model.load_state_dict(best_model_state)
    return model
#

def set_random_seeds(seed):
    # Set seeds for reproducability in randomness
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#

def plot_metrics(
        ax_loss,
        ax_ious,
        train_losses,
        val_losses,
        train_ious,
        val_ious,
        num_classes,
        class_labels=None
    ):

    # Clear previous plots
    ax_loss.clear()
    ax_ious.clear()

    # Plot losses
    ax_loss.plot(train_losses, label='Training Loss')
    ax_loss.plot(val_losses, label='Validation Loss')
    ax_loss.set_title('Losses')
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss')
    ax_loss.legend()
    ax_loss.set_ylim(0, 1)
    ax_loss.set_yticks(np.arange(0, 2.1, 0.1))

    # Plot IoUs for all classes
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(num_classes)]
    
    for c in range(num_classes):
        ax_ious.plot(train_ious[c], label=f'Training IoU {class_labels[c]}')
        ax_ious.plot(val_ious[c], linestyle='--', label=f'Validation IoU {class_labels[c]}')

    ax_ious.set_title('IoUs per class')
    ax_ious.set_xlabel('Epoch')
    ax_ious.set_ylabel('IoU')
    ax_ious.legend(loc='lower right')
    ax_ious.set_ylim(0, 1)
    ax_ious.set_yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.pause(0.1) # pause to allow plot to update
#

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logger.error(f'Error occurred: {e}')
        logger.error(traceback.format_exc())
