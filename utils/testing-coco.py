import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pycocotools.coco import COCO
from matplotlib.patches import Circle

# Define the path to the COCO dataset
data_dir = 'path/to/coco'
ann_file = os.path.join(data_dir, 'annotations', 'person_keypoints_train2017.json')
img_dir = os.path.join(data_dir, 'train2017')

# Initialize COCO API for keypoints annotations
coco = COCO(ann_file)

# Get all image IDs containing persons
cat_ids = coco.getCatIds(catNms=['person'])
img_ids = coco.getImgIds(catIds=cat_ids)

# Select a random image
img_id = img_ids[np.random.randint(0, len(img_ids))]
img_info = coco.loadImgs(img_id)[0]

# Load image
img_path = os.path.join(img_dir, img_info['file_name'])
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Load annotations
ann_ids = coco.getAnnIds(imgIds=img_info['id'], catIds=cat_ids, iscrowd=False)
anns = coco.loadAnns(ann_ids)

# Define the keypoints indices
keypoints_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                   'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                   'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

# Define keypoints connectivity (for drawing skeleton)
skeleton = [[1, 2], [1, 0], [2, 4], [4, 6], [3, 5], [5, 7], [6, 8],
            [7, 9], [8, 10], [5, 11], [6, 12], [11, 13], [12, 14],
            [13, 15], [14, 16]]

# Visualize image and keypoints
plt.imshow(img)
plt.axis('off')

ax = plt.gca()

for ann in anns:
    keypoints = np.array(ann['keypoints']).reshape(-1, 3)
    
    # Draw keypoints
    for i, (x, y, v) in enumerate(keypoints):
        if v > 0:  # v=0: not labeled, v=1: labeled but not visible, v=2: labeled and visible
            circle = Circle((x, y), radius=3, color='red' if v == 2 else 'yellow', fill=True)
            ax.add_patch(circle)
            plt.text(x, y, keypoints_names[i], color="blue", fontsize=8, ha='right', va='bottom')
    
    # Draw skeleton
    for sk in skeleton:
        sk_pts = keypoints[sk, :2]
        if np.all(sk_pts[0] > 0) and np.all(sk_pts[1] > 0):  # both keypoints are labeled
            plt.plot(sk_pts[:, 0], sk_pts[:, 1], color='green')

plt.show()
