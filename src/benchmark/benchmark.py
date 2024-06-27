from mmpose.evaluation.metrics import PCKAccuracy

import numpy as np
from mmengine.structures import InstanceData

def test():
    '''PCK Accuracy example code'''
    num_keypoints = 15
    keypoints = np.random.random((1, num_keypoints, 2)) * 10
    gt_instances = InstanceData()
    gt_instances.keypoints = keypoints
    gt_instances.keypoints_visible = np.ones(
        (1, num_keypoints, 1)).astype(bool)
    gt_instances.bboxes = np.random.random((1, 4)) * 20
    pred_instances = InstanceData()
    pred_instances.keypoints = keypoints
    data_sample = {
        'gt_instances': gt_instances.to_dict(),
        'pred_instances': pred_instances.to_dict(),
    }
    data_samples = [data_sample]
    data_batch = [{'inputs': None}]
    pck_metric = PCKAccuracy(thr=0.5, norm_item='bbox')
        # : UserWarning: The prefix is not set in metric class PCKAccuracy.
    pck_metric.process(data_batch, data_samples)
    pck_metric.evaluate(1)
    print('hi')

def main():
    # testing the PCKAccuracy function
    test()
if __name__ == '__main__':
    main()