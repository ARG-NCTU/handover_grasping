# AUTOGENERATED! DO NOT EDIT! File to edit: 07_other_algorithm.ipynb (unless otherwise specified).

__all__ = ['GGCNN', 'ConvNet']

# Cell
import torch
import torch.nn as nn
import torchvision
import copy
import cv2
import sys
import numpy as np
from PIL import Image
from PIL import ImageDraw
import simplejson as json

# Rosenberger
# ref:https://github.com/patrosAT/h2r_handovers
sys.path.append('/home/arg/ggcnn_humanseg_ros/src/helper_ggcnn')
from ggcnn import predict
from pre_processing import Preparations

# DOPE
# ref:https://github.com/NVlabs/Deep_Object_Pose
sys.path.append('/home/arg/Deep_Object_Pose/scripts/train2')
sys.path.append('/home/arg/Deep_Object_Pose/scripts/train2/inference')
sys.path.append('/home/arg/Deep_Object_Pose/src/dope/inference')
from cuboid import Cuboid3d
from cuboid_pnp_solver import CuboidPNPSolver
from inference import DopeNode, Draw
from detector import ModelData, ObjectDetector, DopeNetwork


# ref:https://github.com/patrosAT/ggcnn_humanseg_ros/blob/master/src/helper_ggcnn/ggcnn.py
class GGCNN():
    def __init__(self):
        self.prep = Preparations()
        self.scale = 300
        self.out_height = 480
        self.out_width = 640


    def largest_indices(self, array, n):
        """Returns the n largest indices from a numpy array.

        This function return top-n index from given numpy array.

        Args:
            array (ndarray) : source array.
            n (int) : num of index to return.

        Returns:
            index

        """
        flat = array.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        indices = indices[np.argsort(-flat[indices])]

        return np.unravel_index(indices, array.shape)

    def angle_translater(self, angle, idx):
        """Remap angle.

        This function will get a specific index in pixel-wise angle numpy array and transfer from radius to degrees.

        Args:
            angle (ndarray) : angle in radius.
            idx (list) : target index of angle array.

        Returns:
            angle (degrees)

        """
        angle = (angle + np.pi/2) % np.pi - np.pi/2
        angle = angle[idx] *180/np.pi

        return angle


    def pred_grasp(self,depth, depth_nan, mask_body, mask_hand, bbox):
        """Get pixel-wise prediction result of grasping point and angle."""
        depth_bg, mask_bg = self.prep.prepare_image_mask(depth=depth, depth_nan=depth_nan,
                                                    mask_body=mask_body, mask_hand=mask_hand,
                                                    dist_obj=bbox[4], dist_ignore=1.0,
                                                    grip_height=0.08)

        depth_ggcnn = cv2.resize(depth_bg,(self.scale,self.scale))
        mask_ggcnn = cv2.resize(mask_bg,(self.scale,self.scale))

        points, angle, width_img, _ = predict(depth=depth_ggcnn, mask=mask_ggcnn,
                                            crop_size=self.scale, out_size=self.scale,
                                            crop_y_offset=40, filters=(2.0, 2.0, 2.0))

        return points, angle, width_img

    def get_grasp_pose(self, points, angle, width_img, top_n = 1):
        """Get Top-N prediction result

        This function will return top-n grasping parameter x, y, theta.

        Args:
            points (ndarray) : pixel-wise grasping point prediction by ggcnn.
            angle (ndarray) : pixel-wise grasping angle prediction by ggcnn.
            width_img (ndarray) : pixel-wise width_img prediction by ggcnn.
            top_n (int) : num of Top-N.

        Returns:
            list of Top-N result:[x, y, theta]
        """
        best_g = self.largest_indices(points, top_n)
        out_list = []
        for i in range(top_n):
            best_g_unr = (best_g[0][i], best_g[1][i])
            Angle = self.angle_translater(angle, best_g_unr)
            resize_point = [best_g_unr[0]*(self.out_height/self.scale), best_g_unr[1]*(self.out_width/self.scale)]

            x = int(resize_point[1])
            y = int(resize_point[0])
            theta = Angle

            out_list.append([x, y, theta])

        return out_list

# ref:https://github.com/andyzeng/arc-robot-vision
class ConvNet(nn.Module):
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        self.color_trunk = torchvision.models.resnet101(pretrained=True)
        del self.color_trunk.fc, self.color_trunk.avgpool, self.color_trunk.layer4
        self.depth_trunk = copy.deepcopy(self.color_trunk)
        self.conv1 = nn.Conv2d(2048, 512, 1)
        self.conv2 = nn.Conv2d(512, 128, 1)
        self.conv3 = nn.Conv2d(128, n_classes, 1)
    def forward(self, color, depth):
        # Color
        color_feat_1 = self.color_trunk.conv1(color) # 3 -> 64
        color_feat_1 = self.color_trunk.bn1(color_feat_1)
        color_feat_1 = self.color_trunk.relu(color_feat_1)
        color_feat_1 = self.color_trunk.maxpool(color_feat_1)
        color_feat_2 = self.color_trunk.layer1(color_feat_1) # 64 -> 256
        color_feat_3 = self.color_trunk.layer2(color_feat_2) # 256 -> 512
        color_feat_4 = self.color_trunk.layer3(color_feat_3) # 512 -> 1024
        # Depth
        depth_feat_1 = self.depth_trunk.conv1(depth) # 3 -> 64
        depth_feat_1 = self.depth_trunk.bn1(depth_feat_1)
        depth_feat_1 = self.depth_trunk.relu(depth_feat_1)
        depth_feat_1 = self.depth_trunk.maxpool(depth_feat_1)
        depth_feat_2 = self.depth_trunk.layer1(depth_feat_1) # 64 -> 256
        depth_feat_3 = self.depth_trunk.layer2(depth_feat_2) # 256 -> 512
        depth_feat_4 = self.depth_trunk.layer3(depth_feat_3) # 512 -> 1024
        # Concatenate
        feat = torch.cat([color_feat_4, depth_feat_4], dim=1) # 2048
        feat_1 = self.conv1(feat)
        feat_2 = self.conv2(feat_1)
        feat_3 = self.conv3(feat_2)
        return nn.Upsample(scale_factor=2, mode="bilinear")(feat_3)