from data_format.coco_dataset.CocoImageLoader import COCOLoader, eval_COCOLoader
from data_format.CocoVideoLoader import COCOVideoLoader
from einops import rearrange
import torch

from data_format.AffineTransform import preprocess_video_data
class eval_COCOVideoLoader(COCOVideoLoader):
    '''A dataformat class for the evaluation dataset of the COCO dataset
    '''
    def __getitem__(self, index):
        initial_image, joint, bbox, mask, image_id = self.image_data[index]
        image = initial_image.detach().clone() # okay, instead of passing the initial image, i'll pass the index
        # width, height
        original_size = torch.tensor([image.shape[2], image.shape[1]])  # Assuming the original size is (height, width)

        # making them all batch size = 1
        # image = image.unsqueeze(0)
        image = rearrange(image, '(d c) h w -> d h w c', d=1)
        joint = joint.unsqueeze(0)
        bbox = bbox.unsqueeze(0)
        #     # some of the bbox have width, and height 0!!!! that means there is nothing in it... (so let me just ignore them in COCOImageLoader)
        processed_image, joint = preprocess_video_data(image.numpy(), bbox.numpy(), joint.numpy(), (self.tensor_width, self.tensor_height), self.min_norm)
        # technically, I have depth = 1... do it's like a one frame video.
        processed_image = rearrange(processed_image, 'd c h w -> c d h w')

        # check if all the joint values are between -1 and 1
        if self.config['full_debug'] and not torch.all((joint >= -1) & (joint <= 1)):
            print("Error, some of the normalized values are not between -1 and 1")
        
        #! TODO fix later: quick patching for evaluation
        # # this means none of the joints in the image are being used, so mAP would be falsely 0.
        if (mask == 0).all():
            # print("This value does not have any relevant input to be used")
            # try a random one lol
            import random
            return self[random.randint(0, len(self) - 1)]
        return processed_image, joint, mask, image_id, original_size, bbox, index