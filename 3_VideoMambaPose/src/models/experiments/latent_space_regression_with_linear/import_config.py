import yaml
import os


def open_config(file_name, folder_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/configs/latent_space_regression'):
    with open(os.path.join(folder_path, file_name), "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful, here are the characteristics of your model: ")
    print(data)
    return data


if __name__ == '__main__':
    open_config('Resized_testing_heatmap_beluga.yaml')
