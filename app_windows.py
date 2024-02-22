from flask import Flask, render_template, request, send_from_directory
import os
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from model import Conv2d, MCNN
from MYDataset import MyDataset, aug_train, aug_val
from sklearn.model_selection import train_test_split

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
        file.save(filename)
        
        # Load the uploaded image
        im = cv2.imread(filename, cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

        # Convert the image to float and normalize it to [0, 1]
        im = im.astype(np.float32) / 255.0

        # Convert the image to a PyTorch tensor and add a batch dimension
        im_tensor = torch.from_numpy(im).unsqueeze(0)  # Only add one dimension for the batch size

        # Pass the image through the model
        model = MCNN(3e-4)
        model.load_state_dict(torch.load('mcnn_model.pth'))
        output = model(im_tensor)

        # Detach the output tensor from the computation graph and remove the batch dimension
        output_image = output.detach().squeeze(0)

        # Display the model's output
        output_image_np = output_image.cpu().numpy()  # convert tensor to numpy array
        plt.figure(figsize=(6, 6))
        plt.imshow(output_image_np)
        plt.title('Model Output')

        # Save the figure
        plt.savefig(os.path.join(app.config['UPLOAD_FOLDER'], 'model_output.png'))

        return render_template('index.html', filename1='uploaded_image.png', filename2='model_output.png')

    return render_template('index.html')

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
