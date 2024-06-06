'''
Helper methods for loading and parsing KITTI data.

Date: April 2020
Ref:https://github.com/charlesq34/frustum-pointnets/tree/master/kitti
'''

from __future__ import print_function

import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3


class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)

        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        # self.predict_dir = os.path.join(self.split_dir, 'predict_64')

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = os.path.join(self.image_dir, '%06d.jpg'%(idx))
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, idx):
        assert(idx<self.num_samples) 
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)
    
    '''
    def get_pred_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training') 
        label_filename = os.path.join(self.predict_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)
    ''' 
    def get_depth_map(self, idx):
        pass

    def get_top_down(self, idx):
        pass

class kitti_object_video(object):
    ''' Load data for KITTI videos '''
    def __init__(self, img_dir, lidar_dir, calib_dir):
        self.calib = utils.Calibration(calib_dir, from_video=True)
        self.img_dir = img_dir
        self.lidar_dir = lidar_dir
        self.img_filenames = sorted([os.path.join(img_dir, filename) \
            for filename in os.listdir(img_dir)])
        self.lidar_filenames = sorted([os.path.join(lidar_dir, filename) \
            for filename in os.listdir(lidar_dir)])
        print(len(self.img_filenames))
        print(len(self.lidar_filenames))
        #assert(len(self.img_filenames) == len(self.lidar_filenames))
        self.num_samples = len(self.img_filenames)

    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = self.img_filenames[idx]
        return utils.load_image(img_filename)

    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = self.lidar_filenames[idx]
        return utils.load_velo_scan(lidar_filename)

    def get_calibration(self, unused):
        return self.calib

def viz_kitti_video():
    video_path = os.path.join(ROOT_DIR, 'dataset/2011_09_26/')
    dataset = kitti_object_video(\
        os.path.join(video_path, '2011_09_26_drive_0023_sync/image_02/data'),
        os.path.join(video_path, '2011_09_26_drive_0023_sync/velodyne_points/data'),
        video_path)
    print(len(dataset))
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, draw_pred_boxes3d
    for i in range(len(dataset)):
        img = dataset.get_image(0)
        pc = dataset.get_lidar(0)
        Image.fromarray(img).show()
        draw_lidar(pc)
        raw_input()
        pc[:,0:3] = dataset.get_calibration().project_velo_to_rect(pc[:,0:3])
        draw_lidar(pc)
        raw_input()
    return

def show_image_with_boxes(img, objects, calib, show3d=True):
    ''' Show image with 2D bounding boxes '''
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox
    for obj in objects:
        if obj.type=='DontCare':continue
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)),
            (int(obj.xmax),int(obj.ymax)), (0,255,0), 2)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d)
    cv2.imwrite('temp.jpg',img2)
    # if show3d:
    #     Image.fromarray(img2).show()
    # else:
    #     Image.fromarray(img1).show()

def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo

def show_lidar_with_boxes(pc_velo, objects, calib,
                          img_fov=False, img_width=None, img_height=None): 
    ''' Show all LiDAR points.
        Draw 3d box in LiDAR point cloud (in velo coord system) '''
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d, draw_pred_boxes3d

    print(('All point num: ', pc_velo.shape[0]))
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    if img_fov:
        pc_velo = get_lidar_in_image_fov(pc_velo, calib, 0, 0,
            img_width, img_height)
        print(('FOV point num: ', pc_velo.shape[0]))
    
    draw_lidar(pc_velo, fig=fig)

    for obj in objects:
        if obj.type=='Car' or obj.type=='Van' or obj.type=='Truck':
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1,y1,z1 = ori3d_pts_3d_velo[0,:]
            x2,y2,z2 = ori3d_pts_3d_velo[1,:]
            draw_gt_boxes3d([box3d_pts_3d_velo], fig=fig)
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
                tube_radius=None, line_width=1, figure=fig)
    '''
    for obj in obj_pred:
        if obj.type=='Car' or obj.type=='Van' or obj.type=='Truck':
            # Draw 3d bounding box
            box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) 
            box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
            # Draw heading arrow
            ori3d_pts_2d, ori3d_pts_3d = utils.compute_orientation_3d(obj, calib.P)
            ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
            x1,y1,z1 = ori3d_pts_3d_velo[0,:]
            x2,y2,z2 = ori3d_pts_3d_velo[1,:]
            draw_pred_boxes3d([box3d_pts_3d_velo], fig=fig)
            mlab.plot3d([x1, x2], [y1, y2], [z1,z2], color=(0.5,0.5,0.5),
                tube_radius=None, line_width=1, figure=fig)
    '''
    mlab.show()

def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)

    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255

    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).show() 
    return img

def show_lidar_topview(pc_velo, objects, calib):
      # 设置鸟瞰图范围
    side_range = (-60, 60)  # 左右距离
    fwd_range = (0, 140.8)  # 后前距离
    
    x_points = pc_velo[:, 0]
    y_points = pc_velo[:, 1]
    z_points = pc_velo[:, 2]
    
    # 获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten()
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    
    res = 0.1  
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    # print(x_img.min(), x_img.max(), y_img.min(), y_img.max()) 
    
    # 填充像素值
    height_range = (-3, 0.0)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])
     
    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)
    
    
    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    image = cv2.cvtColor(im,cv2.COLOR_GRAY2BGR)
    # x = np.arange(0,102.4+0.4,0.4)
    # for cood in x:
    #     point1 = [cood,-60]
    #     point2 = [cood,60]
    #     cv2.line(image, (int(60/res), int(cood/res)),(int(-60/res), int(cood/res)), (0, 255, 0), 1, lineType=cv2.LINE_AA)
    cv2.imwrite("BEV.jpg",image)




    # im2 = Image.fromarray(im)
    # im2.save('BEV.png')
    # im2.show()

def show_lidar_topview_with_boxes(img, objects, calib, color):
    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    boxes3d = [bbox3d(obj) for obj in objects if obj.type == "Car"]
    gt = np.array(boxes3d)
    img = utils.draw_box3d_label_on_bev(img, gt, scores=None, thickness=2, color = color)

    boxes3d = [bbox3d(obj) for obj in objects if obj.type == "Pedestrian"]
    gt = np.array(boxes3d)
    img = utils.draw_box3d_label_on_bev(img, gt, scores=None, thickness=2, color=color)

    boxes3d = [bbox3d(obj) for obj in objects if obj.type == "Cyclist"]
    gt = np.array(boxes3d)
    img = utils.draw_box3d_label_on_bev(img, gt, scores=None, thickness=2, color=color)

    img = Image.fromarray(img)
    img.save('BEV with boxes.png')





    '''
    # pred
    if objects_pred is not None:
        boxes3d = [bbox3d(obj) for obj in objects_pred if obj.type != "DontCare"]
        gt = np.array(boxes3d)
        lines = [obj.type for obj in objects_pred if obj.type != "DontCare"]
        top_image = utils.draw_box3d_on_top(
            top_image, gt, text_lables=lines, scores=None, thickness=1, is_gt=False
        )
    '''

