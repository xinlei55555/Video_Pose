import os
import requests
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

# Define the paths to the dataset
data_dir = '/home/xinleilin/Projects/Video_Pose/data/COCO-Pose/coco'
annotation_file = os.path.join(data_dir, 'annotations/person_keypoints_val2017.json')
image_dir = os.path.join(data_dir, 'images/val2017')

# Initialize the COCO API for person keypoints
coco = COCO(annotation_file)

# Get all image IDs and load an image
image_ids = coco.getImgIds()
image_info = coco.loadImgs(image_ids[0])[0]
image_path = os.path.join(image_dir, image_info['file_name'])

# Load and display the image
image = Image.open(image_path)
plt.imshow(image)
plt.axis('off')

# Load the annotations for the image
annotation_ids = coco.getAnnIds(imgIds=image_info['id'], iscrowd=False)
annotations = coco.loadAnns(annotation_ids)

# Draw keypoints and bounding boxes
fig, ax = plt.subplots(1)
ax.imshow(image)

for annotation in annotations:
    # Draw bounding box
    bbox = annotation['bbox']
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=1, edgecolor='r', facecolor='none')
    ax.add_patch(rect)

    # Draw keypoints
    keypoints = annotation['keypoints']
    for i in range(0, len(keypoints), 3):
        x, y, v = keypoints[i], keypoints[i+1], keypoints[i+2]
        if v > 0:
            ax.plot(x, y, 'bo')
            if v == 2:
                ax.plot(x, y, 'bo', markersize=5)
            elif v == 1:
                ax.plot(x, y, 'bo', markersize=3)

plt.show()
