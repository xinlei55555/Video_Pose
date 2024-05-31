import yaml

def open_config(file_path='/home/linxin67/projects/def-btaati/linxin67/Projects/MambaPose/Video_Pose/3_VideoMambaPose/configs/heatmap/parallel_heatmap_beluga.yaml'):
    with open(file_path, "r") as yamlfile:
        data = yaml.load(yamlfile, Loader=yaml.FullLoader)
    print("Read successful, here are the characteristics of your model: ")
    print(data)
    return data


if __name__ == '__main__':
    open_config()