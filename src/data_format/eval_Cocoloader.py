from data_format.coco_dataset.COCOImageLoader import COCOLoader
from data_format.CocoVideoLoader import COCOVideoLoader

class eval_COCOLoader(COCOLoader):
    '''A dataformat class for the evaluation dataset of the image COCO Loader'''
    def __getitem__(self, index):
        # calling the __getitem__ from the parent class
        # captures the value from the parent class
        image, keypoints, bbox, mask = super().__getitem__(index)

        # redefining the missing variables.
        image_id = self.image_ids[self.new_image_ids[index]]

        # then adds lines
        return image, image_id


class eval_COCOVideoLoader(COCO_VideoLoader):
    '''A dataformat class for the evaluation dataset of the COCO dataset
    '''
    def __getitem__(self, index):
        image, joint, bbox, mask, image_id = self.image_data[index]
        # making them all batch size = 1
        # image = image.unsqueeze(0)
        image = rearrange(image, '(d c) h w -> d h w c', d=1)
        joint = joint.unsqueeze(0)
        bbox = bbox.unsqueeze(0)
        #     # some of the bbox have width, and height 0!!!! that means there is nothing in it... (so let me just ignore them in COCOImageLoader)
        image, joint = preprocess_video_data(image.numpy(), bbox.numpy(), joint.numpy(), (self.tensor_width, self.tensor_height), self.min_norm)
        # technically, I have depth = 1... do it's like a one frame video.
        image = rearrange(image, 'd c h w -> c d h w')

        # check if all the joint values are between -1 and 1
        if self.config['full_debug'] and not torch.all((joint >= -1) & (joint <= 1)):
            print("Error, some of the normalized values are not between -1 and 1")

        return [image, joint, mask, image_id]