''' 
Date: May 2020
Ref:https://github.com/charlesq34/frustum-pointnets/tree/master/kitti
'''
from __future__ import print_function

import os
import sys
import cv2
import random
import os.path
import shutil
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
from kitti_object import *
from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

def visualization():
    import mayavi.mlab as mlab
    dataset = kitti_object('/workspace/mnt/storage/guangcongzheng/zju_wwj_backup/dair-v2x-i-kitti') 
    data_idx = 520

    # Load data from dataset
    objects = dataset.get_label_objects(data_idx)
    # print("There are %d objects.", len(objects))
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
    pc_velo = dataset.get_lidar(data_idx)[:,0:4] # (x, y, z)
    calib = dataset.get_calibration(data_idx)

    # # Show image and Draw 2d and 3d boxes on image
    # print(' ------------ show image without bounding box -------- ')
    # Image.fromarray(img).show()
    # # raw_input()
    
    # # Draw 2d boxed on image
    # print(' ------------ show image with 2D bounding box -------- ')
    # show_image_with_boxes(img, objects, calib, False)
    # # raw_input()

    # print(' ------------ show image with 3D bounding box ------- ')
    # show_image_with_boxes(img, objects, calib, True)
    # raw_input()

    # # Visualize LiDAR points on images
    # print(' ----------- LiDAR points projected to image plane -- ')
    # show_lidar_on_image(pc_velo, img, calib, img_width, img_height)
    # # raw_input()

    # # print(' ------------ LiDAR points that in imageFov-------- ')
    # imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo, calib, 0, 0, img_width, img_height, True)
    # print('imgfov_pc_velo.shape is: ', imgfov_pc_velo.shape)
    # draw_lidar(imgfov_pc_velo, show=True)
    #  # raw_input()

    # # Show all LiDAR points.
    # print(' ---------- LiDAR points in velodyne coordinate ----- ')
    # draw_lidar(pc_velo, show=True)
    # # raw_input()

    # # Draw 3d box in LiDAR point cloud
    # print(' ---------- LiDAR points with 3D boxes in velodyne coordinate ----- ')
    # show_lidar_with_boxes(pc_velo, objects, calib,  False, img_width, img_height)
    # #raw_input()

    # Draw BEV
    # print('------------------ BEV of LiDAR points -----------------------------')
    # show_lidar_topview(pc_velo, objects, calib)
    # raw_input()

    print('--------------- BEV of LiDAR points with bobes ---------------------')
    img1 = cv2.imread('000520.png') 
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    show_lidar_topview_with_boxes(img1, objects, calib, color = (0,255,0))


if __name__=='__main__':
    visualization()