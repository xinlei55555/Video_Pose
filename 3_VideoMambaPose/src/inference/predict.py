import sys
import os

# change the system directory
sys.path.append('/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models')
from experiments.heatmap.HeatVideoMamba import HeatMapVideoMambaPose

import torch 
import torch.nn as nn


def load_model(filepath):
    # Create the model
    model = HeatMapVideoMambaPose()
    
    # Load the state dictionary from the .pt file
    model.load_state_dict(torch.load(filepath))
    
    # Set model to evaluation mode
    model.eval()
    
    return model


def inference(model, input_tensor):
    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(input_tensor)
    return output

def get_input_and_label(action, file_name, path='/home/linxin67/scratch/JHMDB'):
    # os.
    pass


if __name__ == "__main__":
    # Define the .pt file path
    model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_22069.0820.pt'
    
    # Load the model from the .pt file
    model = load_model(model_path)

    print(model)
    
    # # Create a sample input tensor (example with batch size 1 and image size 32x32x3)
    # input_tensor, excepted_output = get_input_and_label()
    
    # # Perform inference
    # output = inference(model, input_tensor)
    
    # # Print the output
    # print("Inference output:")
    # print(output)