import torch
from torch.utils.data import DataLoader
from torch import optim
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex, Dice
from torch.utils.tensorboard import SummaryWriter

from src.dataset import PascalVOCSearchDataset, VOC_CLASSES
from model.pspnet import PSPNet
from src.utils import EarlyStopping, transform
from collections import OrderedDict
import shutil
import argparse

import numpy as np
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")



# Training fuction
def train(model, train_dataloader, device, optimizer, epoch, EPOCHS, writer):
    model.train()
    train_progress = tqdm(train_dataloader, colour="cyan")
    all_losses = []
    for idx, img_mask in enumerate(train_progress):
        img = img_mask[0].float().to(device)  # img - B,C,H,W
        mask = img_mask[1].long().to(device)  # label - B,H,W
        y_pred, main_loss, aux_loss = model(img, mask)  # B, H, W | main loss | aux loss

        # Optimizer
        optimizer.zero_grad()
        loss = main_loss + 0.4 * aux_loss
        loss.backward()
        optimizer.step()

        all_losses.append(loss.cpu().item())
        average_loss = np.mean(all_losses)
        # tracking the loss function
        writer.add_scalar("Train/Loss", average_loss, epoch * len(train_dataloader) + idx)

        train_progress.set_description("TRAIN| Epoch: {}/{}| Iter: {}/{} | Loss: {:0.4f} | lr: {:0.4f}".format(
            epoch + 1, EPOCHS, idx, len(train_dataloader), loss, optimizer.param_groups[0]['lr']))


# Evaluate function
def evaluate(model, val_dataloader, device, acc_metric, miou_metric, dice_metric):
    model.eval()
    # list metrics
    all_acc = []
    all_mIOU = []
    all_dice = []

    with torch.no_grad():
        for idx, img_mask in enumerate(val_dataloader):
            img = img_mask[0].float().to(device)
            mask = img_mask[1].long().to(device)  # B W H

            y_pred = model(img, mask)  # B, 21, H, W

            # (B, 21, H, W) -> (B, 1, H, W) -> (B, H, W)
            y_pred = y_pred.argmax(dim=1).squeeze(dim=1)
            acc = acc_metric(y_pred, mask)
            miou = miou_metric(y_pred, mask)
            dice = dice_metric(y_pred, mask)

            all_acc.append(acc.cpu().item())
            all_mIOU.append(miou.cpu().item())
            all_dice.append(dice.cpu().item())
            if idx > 35: break

    # Compute metrics for the epoch
    acc = np.mean(all_acc)
    miou = np.mean(all_mIOU)
    dice = np.mean(all_dice)

    return acc, miou, dice

#load checkpoint pre-trained pspnet model
def LoadCheckpoint(model,checkpoint_path, device ):
    checkpoint = torch.load(checkpoint_path)
    if not torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    current_epoch = checkpoint['epoch']
    pretrained_dict = checkpoint['state_dict']

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        name = k[7:] if k.startswith('module.') else k  # remove `module.`
        new_state_dict[name] = v

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    new_state_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(new_state_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)
    model.to(device)
    return model


def main(learning_rate, batch_size, epochs, num_workers, data_path, tensorboard_path,checkpoint_path, pre_trained):

    num_classes = len(VOC_CLASSES)

    # Create model save directory if it doesn't exist
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    # Creater tensorboard directory
    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path)
    os.mkdir(tensorboard_path)

    # Data augmentation and preprocessing for training and testing
    train_transform, test_transform = transform()

    # Create datasets and dataloaders
    train_dataset = PascalVOCSearchDataset(root=data_path,image_set="train", transform=train_transform, download=False)
    val_dataset = PascalVOCSearchDataset(root= data_path, image_set="val", transform=test_transform, download=False)


    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True,
                                  drop_last=True)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
                                drop_last=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and move it to the appropriate device
    model = PSPNet(layers=50, classes=num_classes).to(device)

    # Initialize optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Best validation IoU for saving the best model
    best_predict = -1
    current_epoch = 0

    # Load checkpoint
    if pre_trained:
        model = LoadCheckpoint(model, checkpoint_path, device)

    # Metrics
    dice_metric = Dice(num_classes=num_classes, average="macro").to(device)
    miou_metric = MulticlassJaccardIndex(num_classes=num_classes, average="macro").to(device)
    acc_metric = MulticlassAccuracy(num_classes=num_classes, average='macro').to(device)

    # Early Stop
    es = EarlyStopping(patience=10, restore_best_weights=False)

    # set up learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,
                                                     threshold=1e-4,min_lr=0)


    # difine Tensorboard
    writer = SummaryWriter(tensorboard_path)

    # Training loop
    for epoch in range(current_epoch, epochs):  # EPOCHS
        train(model, train_dataloader, device, optimizer, epoch, epochs, writer)
        acc, miou, dice = evaluate(model, val_dataloader, device, acc_metric, miou_metric, dice_metric)

        # write in tensorboard
        writer.add_scalar("Test/Acc", acc, epoch)
        writer.add_scalar("Test/mIOU", miou, epoch)
        writer.add_scalar("Test/Dice", dice, epoch)

        # update learning rate
        scheduler.step(1 - miou)

        # Create checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "optimizer_state_dict": optimizer.state_dict(),
            "dice": dice
        }

        # Save last checkpoint
        torch.save(checkpoint, os.path.join(checkpoint_path, "last.h5"))

        # Save best checkpoint based on dice score
        if dice > best_predict:
            torch.save(checkpoint, os.path.join(checkpoint_path, "best.h5"))
            best_predict = dice

        if es(model, 1 - miou):
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

        print("VAL| Acc:{:0.4f}  | mIOU: {:0.4f} | Dice: {:0.4f} | EStop: {}".format(
            acc, miou, dice, es.status))

def get_args():
    # Build arguments
    parser = argparse.ArgumentParser(description="Train PSPNet model")
    # Add arguments with help descriptions
    parser.add_argument("--data_path", "-d", type=str, default="pascal_voc", help="Path to the dataset")
    parser.add_argument("--batch_size", "-b", type=int, default=5, help="Batch size for training")
    parser.add_argument("--num_workers", "-n", type=int, default=4, help="Number of worker threads for data loading")
    parser.add_argument("--image_size", "-i", type=int, default=257, help="Size of the input images")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--lr", "-l", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--log_path", "-p", type=str, default="tensorboard", help="Path to save the log files for TensorBoard")
    parser.add_argument("--checkpoint_path", "-c", type=str, default="trained_model" , help="Path to save model checkpoints")
    parser.add_argument("--pre_trained", "-t", type=bool , default=False , help="Load checkpoint")

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args= get_args()

    # Hyperparameters and paths
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    num_workers = args.num_workers
    data_path = args.data_path
    tensorboard_path = args.log_path
    checkpoint_path = args.checkpoint_path
    pre_trained = args.pre_trained

    main(learning_rate,batch_size,epochs,num_workers,data_path,
            tensorboard_path,checkpoint_path, pre_trained)
