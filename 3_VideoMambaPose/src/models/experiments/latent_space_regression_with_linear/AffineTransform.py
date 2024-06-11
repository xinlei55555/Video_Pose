# '''
# Contains the necessary functions to transform the videos through the bounding boxes with the VitPose style.
# '''

# import cv2
# from torchvision.transforms import functional as F
# from torchvision import transforms
# import numpy as np
# import math
# import torch
# from einops import rearrange

# def get_warp_matrix(theta, size_input, size_dst, size_target):
#     """Calculate the transformation matrix under the constraint of unbiased.
#     Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
#     Data Processing for Human Pose Estimation (CVPR 2020).

#     Args:
#         theta (float): Rotation angle in degrees.
#         size_input (np.ndarray): Size of input image [w, h].
#         size_dst (np.ndarray): Size of output image [w, h].
#         size_target (np.ndarray): Size of ROI in input plane [w, h].

#     Returns:
#         np.ndarray: A matrix for transformation.
#     """
#     theta = np.deg2rad(theta)
#     matrix = np.zeros((2, 3), dtype=np.float32)
#     scale_x = size_dst[0] / size_target[0]
#     scale_y = size_dst[1] / size_target[1]
#     matrix[0, 0] = math.cos(theta) * scale_x
#     matrix[0, 1] = -math.sin(theta) * scale_x
#     matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
#                               0.5 * size_input[1] * math.sin(theta) +
#                               0.5 * size_target[0])
#     matrix[1, 0] = math.sin(theta) * scale_y
#     matrix[1, 1] = math.cos(theta) * scale_y
#     matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
#                               0.5 * size_input[1] * math.cos(theta) +
#                               0.5 * size_target[1])
#     return matrix

# def box2cs(image_size, box):
#     """This encodes bbox(x,y,w,h) into (center, scale)

#     Args:
#         x, y, w, h

#     Returns:
#         tuple: A tuple containing center and scale.

#         - np.ndarray[float32](2,): Center of the bbox (x, y).
#         - np.ndarray[float32](2,): Scale of the bbox w & h.
#     """

#     x, y, w, h = box[:4]
#     input_size = image_size
#     aspect_ratio = input_size[0] / input_size[1]
#     center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

#     if w > aspect_ratio * h:
#         h = w * 1.0 / aspect_ratio
#     elif w < aspect_ratio * h:
#         w = h * aspect_ratio

#     # pixel std is 200.0
#     scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
#     scale = scale * 1.25

#     return center, scale

# def warp_affine_joints(joints, mat):
#     """Apply affine transformation defined by the transform matrix on the
#     joints.

#     Args:
#         joints (np.ndarray[..., 2]): Origin coordinate of joints.
#         mat (np.ndarray[3, 2]): The affine matrix.

#     Returns:
#         np.ndarray[..., 2]: Result coordinate of joints.
#     """
#     joints = np.array(joints)
#     shape = joints.shape
#     joints = joints.reshape(-1, 2)
#     return np.dot(
#         np.concatenate((joints, joints[:, 0:1] * 0 + 1), axis=1),
#         mat.T).reshape(shape)

# def preprocess_video_data(frames, bboxes, joints, out_res=(192, 256)):
#     """
#     Preprocesses the video data.

#     Args:
#         frames: A numpy array with shape (F, H, W, 3)
#         bbox: A numpy array with shape (F, 4)
#             bbox format should be (x, y, w, h)
#         joints: A numpy array with shape (F, J, 2)
#         out_res: Output resolution in (w, h) format. By default is (192, 256)
#     """
#     # print('initial', joints)
#     image_size = np.array(out_res)
#     num_frames = frames.shape[0]

#     new_frames, new_joints = [], []
#     for i in range(num_frames):
#         center, scale = box2cs(image_size, bboxes[i])
#         rotation = 0
#         trans = get_warp_matrix(rotation, center * 2.0,
#                                 image_size - 1.0, scale * 200.0)

#         # perform Affine warping
#         # Inter linear, keeps the image horizontal/vertical lines, but zooms in
#         frames_cropped = cv2.warpAffine(
#             frames[i],
#             trans, (int(image_size[0]), int(image_size[1])),
#             flags=cv2.INTER_LINEAR)
#         frames_cropped = F.to_tensor(frames_cropped)
#         print('output_size_preprocess', frames_cropped.shape)

#         # normalize the RGB data
#         frames_cropped = F.normalize(frames_cropped, mean=[0.485, 0.456, 0.406], std=[
#                         0.229, 0.224, 0.225])

#         new_frames.append(frames_cropped)

#         new_joints.append(
#             torch.from_numpy(warp_affine_joints(
#                 joints[i][:, 0:2].copy(), trans))
#         )

#     new_frames = torch.stack(new_frames)
#     new_joints = torch.stack(new_joints)

#     # print('final', new_joints)

#     return new_frames, new_joints

# def inverse_process_video_data(frame, bbox, joint, input_res):
#     '''
#     This applies the inverse functions of the preprocess_video_data outputs of a given frame
#     Args:
#         frame is a numpy array for the given frame, must be of the format (c w h)
#         bbox is the numpy array for the box information
#         joint is a numpy array containing the joints for the given frame
#         input_res is a tuple containing the final resolution
#     '''
#     image_size = np.array(input_res)
#     print('image size ', frame.shape)
#     center, scale = box2cs(image_size, bbox)
#     rotation = 0

#     print(frame)

#     # denormalization works
#     invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
#                                                      std = [ 1/0.229, 1/0.224, 1/0.225 ]),
#                                 transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
#                                                      std = [ 1., 1., 1. ]),
#                                ])

#     frame = invTrans(frame)
#     frame = rearrange(frame, 'c w h -> h w c')
#     print(frame.shape)

#     # i'll inverse both
#     inv_trans = get_warp_matrix(-rotation, image_size + 1.0, center / 2.0, scale * 200.0)
#     print('The shape of the trans matrix is', np.shape(inv_trans))
#     # inv_tran = np.linalg.inv(trans) # ! doesn't work because tran is not square.

#     # perform inverse warping
#     frame_cropped = cv2.warpAffine(
#         frame.numpy(),
#         inv_trans, (int(image_size[0]), int(image_size[1])),
#         flags=cv2.INTER_LINEAR)
#     print(np.shape(frame_cropped))
#     frame_cropped = F.to_tensor(frame_cropped)

#     # inverse warping for the joints
#     joint = torch.from_numpy(warp_affine_joints(
#                 joint[:, 0:2].numpy().copy(), inv_trans))

#     return frame_cropped, joint


# if __name__ == '__main__':
#     import matplotlib
#     import matplotlib.pyplot as plt
#     # visualizing the result:
#     # Create a synthetic test video and bounding boxes
#     num_frames = 16
#     height, width = 240, 320
#     out_res = (192, 256)

#     # frames = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
#     frames = torch.tensor([cv2.imread('00001.png')]*num_frames).numpy()

#     bboxes = np.array([[50, 40, 100, 150]] * num_frames)
#     joints = np.random.rand(num_frames, 17, 2) * [width, height]

#     # Preprocess the video
#     preprocessed_frames, preprocessed_joints = preprocess_video_data(frames, bboxes, joints, out_res)

#     # Print the shapes of the original and preprocessed frames
#     print(f"Original frames shape: {frames.shape}")
#     print(f"Preprocessed frames shape: {preprocessed_frames.shape}")

#     # Print the sizes of each frame
#     # for i in range(num_frames):
#     #     print(f"Original Frame {i+1} size: {frames[i].shape}")
#     #     print(f"Preprocessed Frame {i+1} size: {preprocessed_frames[i].shape}")

#     # Visualize the original and preprocessed frames
#     fig, axes = plt.subplots(num_frames, 2, figsize=(10, 40))

#     for i in range(num_frames):
#         axes[i, 0].imshow(frames[i])
#         axes[i, 0].set_title(f'Original Frame {i+1}')
#         axes[i, 0].axis('off')

#         preprocessed_frame = preprocessed_frames[i].permute(1, 2, 0).numpy()
#         preprocessed_frame = (preprocessed_frame - preprocessed_frame.min()) / (preprocessed_frame.max() - preprocessed_frame.min())  # Normalize for display
#         axes[i, 1].imshow(preprocessed_frame)
#         axes[i, 1].set_title(f'Preprocessed Frame {i+1}')
#         axes[i, 1].axis('off')

#     # plt.tight_layout()
#     # plt.savefig('comparison.png')  # Save the visualization to a file
#     # plt.close()
#     # num_frames = 16
#     # height, width = 240, 320
#     # out_res = (192, 256)

#     # frames = np.random.randint(0, 255, (num_frames, height, width, 3), dtype=np.uint8)
#     # bboxes = np.array([[50, 40, 100, 150]] * num_frames)
#     # joints = np.random.rand(num_frames, 17, 2) * [width, height]

#     # preprocessed_frames, preprocessed_joints = preprocess_video_data(frames, bboxes, joints, out_res)

#     frame_idx = 0
#     preprocessed_frame = preprocessed_frames[frame_idx]
#     preprocessed_joint = preprocessed_joints[frame_idx]
#     original_frame = frames[frame_idx]

#     inv_frame, inv_joint = inverse_process_video_data(preprocessed_frame, bboxes[frame_idx], preprocessed_joint, (320, 240))

#     # Check if the joint values are the same after inverse processing
#     joints_diff = np.abs(joints - inv_joint.numpy())
#     max_diff = np.max(joints_diff)
#     print(f"Max difference in joint values after inverse processing: {max_diff}")


#     print(f"Original Frame shape: {original_frame.shape}")
#     print(f"Inverse Processed Frame shape: {inv_frame.shape}")

#     fig, axes = plt.subplots(1, 3, figsize=(15, 5))

#     axes[0].imshow(original_frame)
#     axes[0].set_title('Original Frame')
#     axes[0].axis('off')

#     preprocessed_frame_np = preprocessed_frame.permute(1, 2, 0).numpy()
#     preprocessed_frame_np = (preprocessed_frame_np - preprocessed_frame_np.min()) / (preprocessed_frame_np.max() - preprocessed_frame_np.min())
#     axes[1].imshow(preprocessed_frame_np)
#     axes[1].set_title('Preprocessed Frame')
#     axes[1].axis('off')

#     inv_frame_np = inv_frame.permute(1, 2, 0).numpy()
#     inv_frame_np = (inv_frame_np - inv_frame_np.min()) / (inv_frame_np.max() - inv_frame_np.min())
#     axes[2].imshow(inv_frame_np)
#     axes[2].set_title('Inverse Processed Frame')
#     axes[2].axis('off')

#     plt.tight_layout()
#     plt.savefig('inverse_comparison.png')
#     plt.close()

import cv2
from torchvision.transforms import functional as F
from torchvision import transforms
import numpy as np
import math
import torch
from einops import rearrange


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

    # notice that this is for the rotation matrix, and since we multiplied by 2 earlier, and now 0.5, it cancels!
    # as a result, these values are all 0
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              # here, this is the only value that is not 0
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

    # get the coordinates of the center.
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


def preprocess_video_data(frames, bboxes, joints, out_res, min_norm):
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

    new_frames, new_joints = [], []
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

    # normalize the joints with custom normalization
    new_joints = normalize_fn(new_joints, min_norm, out_res[1], out_res[0])

    return new_frames, new_joints


def inverse_process_joint_data(bbox, joint, output_res, frame=False):
    '''
    This applies the inverse functions of the preprocess_video_data outputs of a given frame
    Args:
        # frame is a numpy array for the given frame, must be of the format (c w h)
        bbox is the numpy array for the box information 
        joint is a numpy array containing the joints for the given frame
        input_res is a tuple containing the final resolution
    '''
    # note: output_res must be the same as out_res in preprocess_video_data
    image_size = np.array(output_res)
    center, scale = box2cs(image_size, bbox)
    rotation = 0

    # Calculate the correct inverse transformation matrix
    trans = get_warp_matrix(rotation, center * 2.0,
                            image_size - 1.0, scale * 200.0)

    # Inverse the of the Affine Transform matrix, notice that the output_res must remain the same, even though to not break the joint values.
    inv_trans = cv2.invertAffineTransform(trans)

        # inverse warping for the joints
    joint = torch.from_numpy(warp_affine_joints(
    joint[:, 0:2].numpy().copy(), inv_trans))

    # although usually, I would not be denormalizing the frames.
    if frame: 
        # denormalization
        invTrans = transforms.Compose([
            transforms.Normalize(mean=[0., 0., 0.], std=[
                                1/0.229, 1/0.224, 1/0.225]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1., 1., 1.]),
        ])

        frame = invTrans(frame)
        frame = rearrange(frame, 'c w h -> h w c')


        # perform inverse warping
        frame_cropped = cv2.warpAffine(
            frame.numpy(),
            inv_trans, (int(image_size[0]), int(image_size[1])),
            flags=cv2.INTER_LINEAR)
        frame = F.to_tensor(frame_cropped)


    return frame, joint


def normalize_fn(x, min_norm, h=240.0, w=320.0):
    # between -1 and 1
    if min_norm == -1:
        # x has num_frames, joint_numbers, (x, y)
        x[:, :, 0] = (x[:, :, 0] / (w / 2.0)) - 1.0  # bewteen -1 and 1
        x[:, :, 1] = (x[:, :, 1] / (h / 2.0)) - 1.0
    if min_norm == 0:
        x[:, :, 0] = x[:, :, 0] / w  # bewteen -1 and 1
        x[:, :, 1] = x[:, :, 1] / h
    return x


def denormalize_fn(x, min_norm, h=240.0, w=320.0):
    # actually, you should do denormalization AFTER the loss funciton. so when doing inference.
    if min_norm == -1:
        # x has num_frames, joint_numbers, (x, y)
        x[:, :, 0] = (x[:, :, 0] + 1.0) * (w / 2.0)  # bewteen -1 and 1
        x[:, :, 1] = (x[:, :, 1] + 1.0) * (h / 2.0)
    if min_norm == 0:
        x[:, :, 0] = x[:, :, 0] * w  # bewteen -1 and 1
        x[:, :, 1] = x[:, :, 1] * h

    return x


def denormalize_default(x, h=240.0, w=320.0, scale=1):
    # since the data was normalize with respect to the w, and h, then if I use my new width, h, would it change somethign?
    '''
    Default initial normalizer:
    (4) pos_world is the normalization of pos_img with respect to the frame size and puppet scale,Â
        the formula is as below

        pos_world(1,:,:) = (pos_img(1,:,:)/W-0.5)*W/H    ./scale;
        pos_world(2,:,:) = (pos_img(2,:,:)/H-0.5)       ./scale;
    '''
    x[:, :, 0] = (x[:, :, 0] * scale / (w/h) + 0.5) * w  # bewteen -0.5 and 0.5
    x[:, :, 1] = (x[:, :, 1] * scale + 0.5) * h
    return x


def det_denormalize_values(x_norm, x_init, scale):
    print(x_init.shape, x_norm.shape)
    h = int(x_init[0][1][1].item() / (0.5 + x_norm[0][1][1].item()))
    w = int(x_init[0][1][0].item() * 2 - 2 * x_norm[0][1][0].item() * h)
    # to change later
    w = abs(w)
    h = abs(h)
    return w, h


def inference_yolo_bounding_box(config, video_tensor):
    '''Returns cropped image, with correct padding around the bounding box to the image size.'''
    # step 1: detect the person, and display bounding boxes
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.eval()

    # YOLOv5 expects a batch of images
    results = model(video_tensor.unsqueeze(0))

    # Extract bounding boxes
    # getting the length of the first dimension
    number_frames = video_tensor.shape[0]
    # (xmin, ymin, xmax, ymax) for each frame
    bounding_boxes = torch.zeros((number_frames, 4))
    for det in results.xyxy[0]:
        xmin, ymin, xmax, ymax, conf, cls = det
        if int(cls) == 0 and conf > 0.5:  # Class 0 is 'person' and threshold is 0.5
            bounding_boxes[i] = torch.tensor(
                [xmin.item(), ymin.item(), xmax.item(), ymax.item()])

    return bounding_boxes


# def pad_image_with_box(config, video_tensor, bounding_box, inference, ground_truth_joints = None):
#     '''
#     Note: this function was not used in the final product.
#     Given the bounding boxes and the images, will return the same image with centered,
#     and cropped version of the image, with white padding outside of the person's bounding box
#     If we are in inference, then the ground_truth_joints must also be updated. Else, then it is passed as nothing
#     '''
#     # remember that currently the video_tensor has 320 x 240. you want to pad until 256x192
#     frames_num, C, H, W = video_tensor.shape
#     cropped_frames = torch.zeros(frames_num)
#     padded_video = torch.zeros(frames_numm)

#     target_width = config['image_tensor_width']
#     target_height = config['image_tensor_height']

#     if frames_num != boundinx_box.shape[0]:
#         print('Error, the cropped image and video length do not match!!!!')

#     for i in range(frames_num):
#         frame = video_tensor[i]
#         bbox = bounding_boxes[i]

#         # Extract bounding box coordinates
#         xmin, ymin, xmax, ymax = bounding_box[i]
#         xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)

#         cropped_height = ymax - ymin
#         cropped_width = xmax-xmin

#         # Crop the frame using the bounding box
#         cropped_frame = functional.crop(
#             frame, ymin, xmin, cropped_height, cropped_width)

#         # In case I need it later
#         cropped_frames[i] = cropped_frame

#         # Calculate the required padding
#         padding_left = (target_width - cropped_width) // 2
#         padding_right = target_width - cropped_width - padding_left
#         padding_top = (target_height - cropped_height) // 2
#         padding_bottom = target_height - cropped_height - padding_top

#         # Apply padding
#         padded_image = F.pad(image,
#                              padding=(padding_left, padding_top,
#                                       padding_right, padding_bottom),
#                              fill=padding_color)
#         padded_video[i] = padded_image
#     return padded_video

if __name__ == '__main__':

    import matplotlib
    import matplotlib.pyplot as plt
    # visualizing the result:
    # # Create a synthetic test video and bounding boxes
    num_frames = 16
    height, width = 240, 320
    out_res = (192, 256)

    frames = torch.tensor([cv2.imread('00001.png')]*num_frames).numpy()

    bboxes = np.array([[50, 40, 100, 150]] * num_frames)
    # # joints = np.rand(num_frames, 17, 2) * [width, height]
    joints = np.array([[[100, 120]] * 15] * num_frames)

    # joints = np.array([[[0, 0]] * 15] * num_frames )
    # # Preprocess the video
    preprocessed_frames, preprocessed_joints = preprocess_video_data(
        frames, bboxes, joints, out_res)

    # after_frames, after_jionts = inverse_process_video_data(preprocessed_frames[0], bboxes[0], preprocessed_joints[0], out_res)

    print(preprocessed_joints)

    # Print the shapes of the original and preprocessed frames
    print(f"Original frames shape: {frames.shape}")
    print(f"Preprocessed frames shape: {preprocessed_frames.shape}")

    # Visualize the original and preprocessed frames
    fig, axes = plt.subplots(num_frames, 2, figsize=(10, 40))

    for i in range(num_frames):
        axes[i, 0].imshow(frames[i])
        axes[i, 0].set_title(f'Original Frame {i+1}')
        axes[i, 0].axis('off')

        preprocessed_frame = preprocessed_frames[i].permute(1, 2, 0).numpy()
        preprocessed_frame = (preprocessed_frame - preprocessed_frame.min()) / (
            preprocessed_frame.max() - preprocessed_frame.min())  # Normalize for display
        axes[i, 1].imshow(preprocessed_frame)
        axes[i, 1].set_title(f'Preprocessed Frame {i+1}')
        axes[i, 1].axis('off')

    # Preprocess and inverse process a single frame
    frame_idx = 0
    preprocessed_frame = preprocessed_frames[frame_idx]
    preprocessed_joint = preprocessed_joints[frame_idx]
    original_frame = frames[frame_idx]

    print('preprocess shape', preprocessed_frame.shape)
    # wrong shape! here, its height = 256, and width = 192
    preprocessed_frame = rearrange(preprocessed_frame, 'c w h -> c h w')
    # (320, 240)) # ERROR, don't the big screen output, put the output you wanted intiially, so it returns you th iniitla coordinates!@ Invert Affine matrix handles it for you!!!
    inv_frame, inv_joint = inverse_process_video_data(
        preprocessed_frame, bboxes[frame_idx], preprocessed_joint, out_res)

    # Check if the joint values are the same after inverse processing
    # for i in range(len(list(joints))):
    # print(joints)
    # print(inv_joint)
    # Max difference in joint values after inverse processing: 8.722770417080028e-07!!!! SO GOOD!!!
    joints_diff = np.abs(joints - inv_joint.numpy())
    print(inv_joint)
    max_diff = np.max(joints_diff)
    print(
        f"Max difference in joint values after inverse processing: {max_diff}")

    print(f"Original Frame shape: {original_frame.shape}")
    print(f"Inverse Processed Frame shape: {inv_frame.shape}")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_frame)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    preprocessed_frame_np = preprocessed_frame.permute(2, 1, 0).numpy()
    preprocessed_frame_np = (preprocessed_frame_np - preprocessed_frame_np.min()) / (
        preprocessed_frame_np.max() - preprocessed_frame_np.min())
    axes[1].imshow(preprocessed_frame_np)
    axes[1].set_title('Preprocessed Frame')
    axes[1].axis('off')

    inv_frame_np = inv_frame.permute(1, 2, 0).numpy()
    inv_frame_np = (inv_frame_np - inv_frame_np.min()) / \
        (inv_frame_np.max() - inv_frame_np.min())
    axes[2].imshow(inv_frame_np)
    axes[2].set_title('Inverse Processed Frame')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig('inverse_comparison.png')
    plt.close()
