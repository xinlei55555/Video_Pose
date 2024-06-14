'''
File created in order to test the baseball swing pictures from earlier.
'''
from Affine Transform import *

def read_joints_full_video(path):
    '''
    Given an action and a video returns the joints for each frame

    Args:
        action, video and path are strings indicating the path of the joints.

    Returns 
        a dictionary dict_keys(['__header__', '__version__', '__globals__', 'pos_img', 'pos_world', 'scale', 'viewpoint'])

        Each file is the following dimension:
        (2, 15 (num joints), n_frames)

        First, there are two dimension, which is x, y
        Then, 
        In pos image, each array has n number of values, where n is the number of frames in the video.
    '''
    
    mat = scipy.io.loadmat(
        f'path')
    return mat

def rearranged_joints(path):
    '''
    Args:
        action, video and path are strings indicating the path of the joints.

    Return 
        a torch tensor with num frames, num joints, (x,y) joints.
    '''
    joint_dct = self.read_joints_full_video(path)

    # we will most likely never use pos_world
    # # pos_world was already normalized with respect to the image. (unlike pos_img)
    # if self.normalized and self.default:
    #     torch_joint = torch.tensor(joint_dct['pos_world'])
    #     torch_joint = rearrange(torch_joint, 'd n f->f n d')
    # then use custom normalization
    # elif self.normalized and not self.default:
    torch_joint = torch.tensor(joint_dct['pos_img'])
    torch_joint = rearrange(torch_joint, 'd n f->f n d')
    # torch_joint = normalize_fn(torch_joint, self.config)
    # then no normalization
    # else:
    # torch_joint = torch.tensor(joint_dct['pos_img'])
    # # rearrange for training and normalization.
    # torch_joint = rearrange(torch_joint, 'd n f->f n d')

    if self.config['full_debug']:
        print(f'normalized: {self.normalized}, default: {self.default}')
        print('example joint values', torch_joint[0][0])

    return torch_joint

if __name__ == '__main__':
# first, let's get the bboxes
    