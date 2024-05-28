import sys
import os

# change the system directory
sys.path.append('/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap')
from HeatVideoMamba import HeatMapVideoMambaPose

import torch 
import torch.nn as nn


def load_model(filepath):
    # Create the model
    model = HeatMapVideoMambaPose()
    print(model)
    print(type(model.state_dict()))
    # print(model.state_dict().keys())
    
    # Load the state dictionary from the .pt file
    # model.load_state_dict(torch.load(filepath))
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    # print(checkpoint)
    print(type(checkpoint))
    # model.load_state_dict(checkpoint['model_state_dict'])
    # print(checkpoint.keys())

    print('checking whether they have the same keys, ', model.state_dict().keys()==checkpoint.keys())
    # checking which keys differ
    x = set(model.state_dict().keys()).intersection(set(checkpoint.keys()))
    # print(x)
    print('that was the intersection ^')

    # checking the keys that are not common
    y = set(model.state_dict().keys()).union(set(checkpoint.keys()))

    diff = y.difference(x)

    # print(diff, 'differeneeceee') #{'joints.regressor.2.bias', 'joints.regressor.0.bias', 'joints.regressor.2.weight', 'joints.regressor.0.weight'}

    # just print the ones with joints.
    joint = {val for val in y if ('joint' in val)}
    # print(joint, ' --->>>joint')

    # print(joint.intersection(set(model.state_dict().keys())))
    # print(joint.intersection(set(checkpoint.keys()))) # the joints are in the checkpoint, but not in them odel I am loading?

    # THE EERRRO was because in my joint regressor, I had not put the layesr in the initisqueuealization funcgtion.
    model.load_state_dict(checkpoint) # strict = False makes it so that even though some layer are missing, it will work (although idk why some layesr are missing)

    # Set model to evaluation mode
    model.eval()
    
    return model


def inference(model, input_tensor):
    # Disable gradient computation for inference
    with torch.no_grad():
        output = model(input_tensor)
    return output

# i'll finish the code on my local machine
def get_input_and_label(action, file_name, path='/home/linxin67/scratch/JHMDB'):
    # os.
    pass


if __name__ == "__main__":
    # Define the .pt file path
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_22069.0820.pt'
    # model_path = '/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_27345.4473.pt'
    model_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/src/models/experiments/heatmap/checkpoints/heatmap_8652135.9131.pt'
    
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