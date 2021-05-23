import argparse
import os
import sys
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms.functional as TF
import torch.nn.functional as F

from io import StringIO
from six import BytesIO

# import model
from MPRNet import MPRNet

# accepts and returns numpy data
CONTENT_TYPE = 'application/x-npy'


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    '''
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))
    '''
    
    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MPRNet()

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model_deblurring.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # prep for testing
    model.to(device).eval()

    print("Done loading model.")
    return model

def input_fn(serialized_input_data, content_type):
    print('Deserializing the input data.')
    if content_type == CONTENT_TYPE:
        stream = BytesIO(serialized_input_data)
        return np.load(stream)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)

def output_fn(prediction_output, accept):
    print('Serializing the generated output.')
    if accept == CONTENT_TYPE:
        buffer = BytesIO()
        np.save(buffer, prediction_output)
        return buffer.getvalue(), accept
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)

def predict_fn(input_data, model):
    print('Predicting class labels for the input data...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    use_cuda = False
    if torch.cuda.is_available():
        device = torch.device("cuda")
        data = data.to(device)   
        use_cuda = True
    else:
        device = torch.device("cpu")
    # Process input_data so that it is ready to be sent to our model
    # convert data to numpy array then to Tensor
    # data = torch.from_numpy(input_data.astype('float32'))
    
    img_multiple_of = 8
    data = TF.to_tensor(data).unsqueeze(0).cuda()
    '''
    if use_cuda:
        data = TF.to_tensor(data).unsqueeze(0).cuda()
    else:
        data = TF.to_tensor(data).unsqueeze(0)
    '''
    # Pad the input if not_multiple_of 8
    h,w = data.shape[2], data.shape[3]
    H,W = ((h+img_multiple_of)//img_multiple_of)*img_multiple_of, ((w+img_multiple_of)//img_multiple_of)*img_multiple_of
    padh = H-h if h%img_multiple_of!=0 else 0
    padw = W-w if w%img_multiple_of!=0 else 0
    data = F.pad(data, (0,padw,0,padh), 'reflect')

    # Put model into evaluation mode
    model.eval()
    
    with torch.no_grad():
        out = model(data)

    restored = out[0]
    restored = torch.clamp(restored, 0, 1)

    # Unpad the output
    restored = restored[:,:,:h,:w]

    restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

    return restored