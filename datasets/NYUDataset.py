import os
import sys

import torch
import scipy.io as sio
import numpy as np
from PIL import Image


class NYUDataset(torch.utils.data.Dataset):
    """NYU Dataset."""

    def __init__(self, root_dir, sample_transform, train=False):
        """
        Args:
            root_dir (string): Path to the data.
            sample_transform (callable, optional): Optional transform to be
                applied to the sample.
        """
        self.root_dir = root_dir
        self.sample_transform = sample_transform
        self.num_kp = 21
        self.train = train

        # Load annotation file
        anno_mat = sio.loadmat(os.path.join(self.root_dir, "joint_data.mat"))
        self.annotations2d = anno_mat['joint_uvd'][0]
        self.joint_idxs = [0, 1, 3, 5, 6, 7, 9, 11, 12, 13, 15, 17, 18, 19, 21, 23, 24, 25, 26, 28, 35, 32]

    def __len__(self):
        if self.train:
            return 72757
        else:
            return 8252

    def __getitem__(self, idx):
        depth_name = os.path.join(self.root_dir, 'depth_1_{0:07d}.png'.format(idx+1))

        depth = Image.open(depth_name)
        w, h = depth.size

        # Process depth
        r, g, b = depth.split()
        r = np.asarray(r, np.int32)
        g = np.asarray(g, np.int32)
        b = np.asarray(b, np.int32)
        depth = np.bitwise_or(np.left_shift(g, 8), b)
        depth = np.asarray(depth, np.float32)

        # TODO: Manually selecting input for now
        sample = depth

        kps2d = self.annotations2d[idx, self.joint_idxs]

        # Process and normalize joints
        bbox = self.get_bbox(kps2d)

        sample, padding = self.crop_depth(sample, bbox)

        if self.sample_transform:
            sample = self.sample_transform(sample)

        sample = self.normalize_depth(sample)

        return sample

    def crop_depth(self, img, bbox):
        """Crop the depth image to the bounding box.

        If the cropped image is not square, 0-value padding will be added.

        Args:
            img (float, H x W x D): Depth array.
            bbox (float, 6): Bounding box of the hand in image space.

        Returns:
            Cropped image (float, H_c x W_c x D) and the row and column
            padding size added to the image (int, 2 x 2).
        """
        xstart = bbox[0]
        xend = bbox[1]
        ystart = bbox[2]
        yend = bbox[3]
        zstart = bbox[4]
        zend = bbox[5]

        cropped = img[max(ystart, 0):min(yend, img.shape[0]), max(xstart, 0):min(xend, img.shape[1])].copy()

        # Crop z bound
        mask1 = np.logical_and(cropped < zstart, cropped != 0)
        mask2 = np.logical_and(cropped > zend, cropped != 0)
        cropped[mask1] = zstart
        cropped[mask2] = 0.0

        if cropped.shape[0] > cropped.shape[1]:
            diff = cropped.shape[0] - cropped.shape[1]
            row_pad = [0, 0]
            if diff % 2 == 1:
                col_pad = [int(diff / 2), int(diff / 2) + 1]
            else:
                col_pad = [int(diff / 2), int(diff / 2)]
        else:
            diff = cropped.shape[1] - cropped.shape[0]
            col_pad = [0, 0]
            if diff % 2 == 1:
                row_pad = [int(diff / 2), int(diff / 2) + 1]
            else:
                row_pad = [int(diff / 2), int(diff / 2)]

        return np.pad(cropped, (row_pad, col_pad), mode='constant', constant_values=0), (row_pad, col_pad)

    def get_bbox(self, keypoints, pad=25):
        """Calculates a 3d bounding box.

        Args:
            keypoints (array): 3d keypoints of the hand in either image or 3d
              space.
            pad (int): Amount of padding to add to the bounding box for all
              sides.
        Returns:
            6 values defining the bounding cube.
        """

        joints_min = keypoints.min(0) - pad
        joints_max = keypoints.max(0) + pad
        return np.array([joints_min[0], joints_max[0],
                         joints_min[1], joints_max[1],
                         joints_min[2], joints_max[2]]).astype(np.int)

    def normalize_depth(self, depth_img):
        """Normalize depth image to be in range [-1, 1].
        Returns a clone of the original image.
        """
        norm_img = depth_img.clone()
        bg_mask = (norm_img == 0)
        fg_mask = (norm_img > 0)
        min_val = norm_img[fg_mask].min()
        max_val = norm_img[fg_mask].max()
        norm_img[fg_mask] -= min_val
        norm_img[fg_mask] /= (max_val - min_val)
        norm_img[fg_mask] *= 2.0
        norm_img[fg_mask] -= 1.0
        norm_img[bg_mask] = 1.0

        return norm_img

