import torch
from src.dataset import PascalVOCSearchDataset, VOC_CLASSES, VOC_COLORMAP
from model.pspnet import PSPNet
from train import transform
from src.utils import UnNormalize

import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


def pred_show_image_grid(data_path,model_pth, device, transform, num_classes, num_imgs):
    # Load model checkpoint
    checkpoint = torch.load(model_pth)

    # Initialize and load model state
    model = PSPNet(layers=50, classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    # Load validation dataset
    image_dataset = PascalVOCSearchDataset(root=data_path, image_set="val", download= False, transform=transform)

    images, orig_masks, pred_masks = [], [], []

    for i in range(num_imgs):
        # Randomly select an image and mask
        idx = np.random.randint(len(image_dataset))
        print(idx, end=" ")
        # idx = [899, 390, 1430, 854, 677, 1205, 1089, 1001]
        img, orig_mask = image_dataset[idx]
        # Store original image
        images.append(unorm(img).permute(1, 2, 0))

        # Predict mask
        img = img.float().to(device).unsqueeze(0)
        # (B, 21, H, W) -> (B, 1, H, W) -> (B, H, W)
        pred_mask = model(img).argmax(dim=1).squeeze(0).cpu().numpy()

        # Colorize masks
        color_pred_mask = np.zeros((*pred_mask.shape, 3))
        color_mask = np.zeros((*orig_mask.shape, 3))

        for i, color in enumerate(VOC_COLORMAP):
            color_pred_mask[pred_mask == i] = np.array(color)
            color_mask[orig_mask == i] = np.array(color)

        orig_masks.append(color_mask)
        pred_masks.append(color_pred_mask)

    # Combine images and masks for display
    images.extend(orig_masks)
    images.extend(pred_masks)

    # Plot images and masks
    fig = plt.figure(figsize=(20, 10))
    for i in range(1, 3 * num_imgs + 1):
        fig.add_subplot(3, num_imgs, i)
        plt.imshow(images[i - 1])
    plt.show()


if __name__ == '__main__':

    unorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    # Define the transformation to be applied to the test images
    _, test_transform = transform()

    # Note: You must train the model before using this param
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "trained_model/best.h5"
    num_classes = len(VOC_CLASSES)
    data_path= "pascal_voc"
    num_imgs=8

    # Show images
    pred_show_image_grid(data_path, model_path, device, test_transform, num_classes,num_imgs)