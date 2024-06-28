from data_format.coco_dataset.CocoImageLoader import COCOLoader, eval_COCOLoader
from data_format.CocoVideoLoader import COCOVideoLoader
from einops import rearrange

from data_format.AffineTransform import preprocess_video_data
class eval_COCOVideoLoader(COCOVideoLoader):
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

        return image, joint, mask, image_id