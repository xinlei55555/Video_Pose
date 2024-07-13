'''
Testing the cocoeval against itself
'''
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

import torch 
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from einops import rearrange
import argparse
import sys
import os
import json
import numpy as np

# first import the dataset from data_format
from data_format.eval_Cocoloader import eval_COCOVideoLoader
from data_format.coco_dataset.CocoImageLoader import eval_COCOLoader
from import_config import open_config
from data_format.AffineTransform import denormalize_fn, inverse_process_joint_data

from models.heatmap.HeatVideoMamba import HeatMapVideoMambaPose
from models.HMR_decoder.HMRMambaPose import HMRVideoMambaPose
from models.MLP_only_decoder.MLPMambaPose import MLPVideoMambaPose
from models.HMR_decoder_coco_pretrain.HMRMambaPose import HMRVideoMambaPoseCOCO
from data_format.AffineTransform import box2cs
from inference.visualize_coco import visualize, load_model

import random

def main():
    cocoGt = COCO('/home/xinleilin/Projects/Video_Pose/data/COCO-Pose/coco/annotations/person_keypoints_val2017.json')

    PERSON_CAT_ID = 1
    person_ann_ids = cocoGt.getAnnIds(catIds=[PERSON_CAT_ID])
    person_anns = cocoGt.loadAnns(ids=person_ann_ids)

    # print(person_anns, 'are the person annotations')
    results = []

    # test
    all_ids = set()

    for person_ann in person_anns:
        if person_ann['num_keypoints'] > 0:
            results.append({
                "image_id": person_ann['image_id'],
                "category_id": PERSON_CAT_ID,
                'keypoints': person_ann['keypoints'],
                'score': 1.0,
                'id': person_ann['id']
                # 'bbox': person_ann['bbox']

            })
            # just verifying the actual id to see if many values:
            if person_ann['image_id'] in all_ids:
                print("duplicate id !!!!!")
            all_ids.add(person_ann['image_id'])
            # just verifying the values
            if person_ann['image_id'] == 468965:
                print({
                    "image_id": person_ann['image_id'],
                    "category_id": PERSON_CAT_ID,
                    'keypoints': person_ann['keypoints'],
                    'score': 1.0,
                    'bbox': person_ann['bbox']
                })
                print('here is the full person annotations', person_ann)

    # okay, that does not matter.
    # let's try changing the order of the elements in the list
    # results = sorted(results, key=lambda x: random.random())

            
    with open("outputs/results.txt", "w") as f:
        f.write(json.dumps(results))
        
    cocoDt = cocoGt.loadRes("outputs/results.txt")

    cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == '__main__':
    main()