import os
from PIL import Image

def get_image_sizes_in_folder(folder_path='/home/linxin67/scratch/JHMDB/Rename_Images'):
    total_size = 0
    image_count = 0
    sizes = set()
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
            except Exception as e:
                print(f"Error processing {file}: {e}")
    print('Here are the different sizes: ', size)
    print(f"Total images: {image_count}")
    print(f"Total size of images: {total_size / (1024 * 1024):.2f} MB")

# Example usage
get_image_sizes_in_folder()
