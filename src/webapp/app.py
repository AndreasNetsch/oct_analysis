from flask import Flask, request, render_template
import torch
# from src.dl_model.main_train import build_model
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp

def create_app():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # create app object using Flask class
    app = Flask(__name__, static_folder='static')

    # load the model
    loadpath = os.path.join(current_dir, 'models', 'latest_model.pth')

    encoder = 'efficientnet-b0' # available: b0, b1, b2, b3, b4, b5, b6, b7 (model size increases with number)
    weights = 'imagenet'
    colorchannels = 1
    activation = None
    depth = 5
    model = build_model(encoder, weights, colorchannels, 3, activation, depth)
    model.load_state_dict(torch.load(loadpath))
    model.eval() # put model in inference mode

    # define a route for the endpoint
    # the decorator @app.route() links the decorated function to the endpoint (URL)
    # render_template() function looks for the file in the templates folder
    # '/' is the root URL --> homepage
    @app.route('/')
    def home():
        return render_template('index.html')
    
    # POST: browser sends data to the server
    # redirect to the /predict page with the output
    @app.route('/predict', methods=['POST'])
    def predict():
        # get the image from the request
        img = request.files['img']

        # copy and save image to disk to display later
        img_copy = img
        img_copy_path = os.path.join(current_dir, 'static', 'uploaded_image.png')
        img_copy.save(img_copy_path)

        # preprocess the image, assuming it is a .png file (later .oct file)
        img = preprocess_image(img)

        # make a prediction
        try:
            with torch.no_grad():
                prediction = model(img)
                prediction = torch.argmax(prediction, dim=1).squeeze().numpy()
        except Exception as e:
            print(e)
            return 'An error occurred while making the prediction.'
        
        prediction_exists = prediction.size > 0 # True if prediction is not empty

        # convert the outputs to a plot and save it
        plot_path = os.path.join(current_dir, 'static', 'output_plot.png')
        plot_outputs(img, prediction, plot_path)

        return render_template('index.html', output_plot='/static/output_plot.png', prediction_exists=prediction_exists)
    #
    return app
###
        

def preprocess_image(img):
    # read the image
    img = Image.open(img).convert('L')
    img = np.array(img).astype('float32')
    img -= img.min()
    img /= img.max() # Normalize to [0,1]

    # crop the image
    img = crop(img)

    # convert to tensor
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0) # add batch and channel dimensions

    return img
#

def crop(img: np.ndarray):
    # Crop image and mask to ensure divisibility by 32
    h, w = img.shape

    # height adjustments
    if h % 32 != 0:
        h_mod = h % 32
        h -= h_mod
        img = img[:h, :]

    # width adjustments
    if w % 32 != 0:
        w_mod = w % 32
        w -= w_mod
        img = img[:, :w]

    return img
#

def plot_outputs(img, prediction, plotpath):

    img = img.squeeze().numpy()
    prediction = prediction.squeeze()

    _, axes = plt.subplots(1, 2, figsize=(20, 10))

    # plot original img
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # plot prediction with color map and legend
    axes[1].imshow(prediction, cmap='viridis') # purple to yellow
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    plt.savefig(plotpath, bbox_inches='tight', pad_inches=0.1)
    plt.close()
#

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


if __name__ == '__main__':
    app = create_app()
    app.run()
