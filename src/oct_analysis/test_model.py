
# 3rd party modules
import matplotlib.pyplot as plt
import torch
import segmentation_models_pytorch as smp


# own modules
from src.oct_analysis.ml_datasets import OCTDataset


def main():
    encoder = 'efficientnet-b2' # available: b0, b1, b2, b3, b4, b5, b6, b7 (model size increases with number)
    weights = 'imagenet'
    colorchannels = 1
    class_labels = ['background', 'biofilm', 'optical window', 'membrane', 'spacer']
    n_classes = len(class_labels)
    activation = None
    depth = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(encoder, weights, colorchannels, n_classes, activation, depth)
    model_path = input("Enter the model path: ")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_imgs = r"C:\Users\sx1218\Arbeitsordner\Nextcloud\ML_OCT\iap_projects\images"
    test_masks = r"C:\Users\sx1218\Arbeitsordner\Nextcloud\ML_OCT\iap_projects\segmented_images"
    dataloader_test = OCTDataset(test_imgs, test_masks)

    check = True
    while check:
        check = input("Test inference? (y/n): ").strip().lower()
        if check == "y":
            show_rand_prediction(model, dataloader_test, device)
        elif check == "n":
            check = False
        else:
            print("Invalid input. Please enter 'y' or 'n'.")
        #
    #
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

def show_rand_prediction(model, dataloader, device):

    # Get a random sample from the dataloader
    idx = torch.randint(0, len(dataloader), (1,)).item()
    img, mask = dataloader[idx]
    img = img.to(device).unsqueeze(0)
    mask = mask.to(device)

    with torch.no_grad():
        output = model(img)
        prob = torch.softmax(output, dim=1)
        membrane_prob = prob[0, 1]
        pred = output.argmax(1)

        img = img.cpu().squeeze().numpy()
        mask = mask.cpu().squeeze().numpy()
        pred = pred.cpu().squeeze().numpy()
        membrane_prob = membrane_prob.cpu().squeeze().numpy()

        _, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs = axs.flatten()
        axs[0].imshow(img, cmap='gray')
        axs[0].set_title("Input Image")
        axs[1].imshow(mask, cmap='gray')
        axs[1].set_title("Ground Truth Mask")
        axs[2].imshow(pred, cmap='gray')
        axs[2].set_title("Predicted Mask")
        im = axs[3].imshow(membrane_prob, cmap='viridis', vmin=0, vmax=1)
        plt.colorbar(im, ax=axs[3], label='Probability')
        axs[3].set_title("Membrane Prediction Probability Map")
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()
    #
#


# def test(model, dataloader, loss_fn, iou_metric, device):
#     model.eval()
#     test_loss = 0
#     iou_metric.reset()
#     with torch.no_grad():
#         for i, (imgs, masks) in enumerate(dataloader):
#             imgs = imgs.to(device)
#             masks = masks.to(device)
#             outputs = model(imgs)
#             loss = loss_fn(outputs, masks)
#             test_loss += loss.item()
#             iou_metric(outputs, masks)
#     print(f"Test loss: {test_loss / len(dataloader)}")
#     print(f"Test IoU: {iou_metric.compute()}")
# #

if __name__ == '__main__':
    main()
