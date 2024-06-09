'''
Contains the necessary functions to transform the videos through the bounding boxes with the VitPose style.
'''

import cv2
from torchvision.transforms import functional as F

def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def box2cs(image_size, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    input_size = image_size
    aspect_ratio = input_size[0] / input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.25

    return center, scale

def warp_affine_joints(joints, mat):
    """Apply affine transformation defined by the transform matrix on the
    joints.

    Args:
        joints (np.ndarray[..., 2]): Origin coordinate of joints.
        mat (np.ndarray[3, 2]): The affine matrix.

    Returns:
        np.ndarray[..., 2]: Result coordinate of joints.
    """
    joints = np.array(joints)
    shape = joints.shape
    joints = joints.reshape(-1, 2)
    return np.dot(
        np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
        mat.T).reshape(shape)

def preprocess_video_data(frames, bboxes, joints, out_res=(192, 256)):
    """
    Preprocesses the video data.

    Args:
        frames: A numpy array with shape (F, H, W, 3)
        bbox: A numpy array with shape (F, 4)
            bbox format should be (x, y, w, h)
        joints: A numpy array with shape (F, J, 2)
        out_res: Output resolution in (w, h) format. By default is (192, 256)
    """
    image_size = np.array(out_res)
    num_frames = frames.shape[0]

    new_frames, new_joints = []
    for i in range(num_frames):
        center, scale = box2cs(image_size, bboxes[i])
        rotation = 0
        trans = get_warp_matrix(rotation, center * 2.0,
                                image_size - 1.0, scale * 200.0)

        # perform Affine warping
        frames_cropped = cv2.warpAffine(
            frames[i],
            trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        frames_cropped = F.to_tensor(frames_cropped)

        # normalize the RGB data
        frames_cropped = F.normalize(frames_cropped, mean=[0.485, 0.456, 0.406], std=[
                        0.229, 0.224, 0.225])
        
        new_frames.append(frames_cropped)

        new_joints.append(
            torch.from_numpy(warp_affine_joints(
                joints[i][:, 0:2].copy(), trans))
        )

    new_frames = torch.stack(new_frames)
    new_joints = torch.stack(new_joints)

    return new_frames, new_joints

