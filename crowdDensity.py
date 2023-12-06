import matplotlib
matplotlib.use('Agg')
import base64
from io import BytesIO
from flask_cors import CORS
from matplotlib import pyplot as plt
import numpy as np
import cv2
import pandas as pd
import os 
from datetime import datetime 
import csv
import torch

from src.models.CSRNet import CSRNet


def counter(frame):
    try:
        model = CSRNet()
        PATH = 'https://huggingface.co/muasifk/CSRNet/resolve/main/CSRNet.pth'
        state_dict = torch.hub.load_state_dict_from_url(PATH, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        print('\n Model loaded successfully.. \n')
        # Convert received frame to an image array
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255
        frame = torch.from_numpy(frame).permute(2, 0, 1)

        # Predict with the model
        et = model(frame.unsqueeze(0))
        count = et.sum()
       

        # Visualization using Matplotlib
        out = et.squeeze().detach().numpy()
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6, 3))
        ax0.imshow(frame.permute(1, 2, 0))
        ax1.imshow(out, cmap='jet')
        ax0.set_title('People Counter')
        ax1.set_title(f'Count = {count:.0f}')
        ax0.axis("off")
        ax1.axis("off")
        plt.tight_layout()

        # Save the plot as an image
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plt.close()  # Close the plot

        # Convert the buffer to bytes and encode it as base64
        encoded_image_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return encoded_image_data

    except Exception as e:
        print(f"Error: {e}")
        return None, None

