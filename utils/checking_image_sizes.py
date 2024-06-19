import os
from PIL import Image

def get_image_sizes_in_folder(action='', folder_name='', folder_path='/home/linxin67/scratch/JHMDB/Rename_Images'):
    folder_path = folder_path+'/'+action+'/'+folder_name
    total_size = 0
    image_count = 0
    sizes = set()

    max_size = (0, 0)
    min_size = (float('inf'), float('inf'))
    max_image = ''
    min_image = ''

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    print(f"Image: {file} - Size: {width}x{height}")
                    sizes.add((width, height))
                    total_size += os.path.getsize(file_path)
                    image_count += 1

                    # Track the largest image
                    if width * height > max_size[0] * max_size[1]:
                        max_size = (width, height)
                        max_image = file_path

                    # Track the smallest image
                    if width * height < min_size[0] * min_size[1]:
                        min_size = (width, height)
                        min_image = file_path

            except Exception as e:
                print(f"Error processing {file}: {e}")

    print('Here are the different sizes: ', sizes)
    print(f"Total images: {image_count}")
    print(f"Total size of images: {total_size / (1024 * 1024):.2f} MB")
    print(f"Largest image: {max_image} - Size: {max_size[0]}x{max_size[1]}")
    print(f"Smallest image: {min_image} - Size: {min_size[0]}x{min_size[1]}")

# Example usage
get_image_sizes_in_folder(folder_path='/home/xinleilin/Projects/Video_Pose/data/COCO-Pose/coco/images/val2017')
