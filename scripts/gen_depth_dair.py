import os
import mmcv
from tqdm import tqdm
import json
import numpy as np
import open3d as o3d

def read_json(path_json):
    with open(path_json, "r") as load_f:
        my_json = json.load(load_f)
    return my_json

def get_P(path):
    my_json = read_json(path)
    P = np.array(my_json["cam_K"]).reshape(3,3)
    return P

def get_velo2cam(path):
    my_json = read_json(path)
    t_velo2cam = np.array(my_json["translation"])
    r_velo2cam = np.array(my_json["rotation"])
    return r_velo2cam, t_velo2cam

def read_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    pts = np.asarray(pcd.points)
    return pts

def project_cam2img(pt, cam_to_img):

    point = np.array(pt)

    point = np.dot(cam_to_img, point)

    point = point[:2]/point[2]
    point = point.astype(np.int64)

    return point

def obtain_point_depth(points, camera_intrinsic, r_velo2cam, t_velo2cam):
    H = 1080
    W = 1920
    
    pts_cam = r_velo2cam.dot(points.T) + t_velo2cam
    pts_pix = project_cam2img(pts_cam, camera_intrinsic)
    
    mask = np.ones(pts_cam.shape[1], dtype=bool)
    mask = np.logical_and(mask, pts_cam[2] > 0)
    mask = np.logical_and(mask, pts_pix[0, :] > 1)
    mask = np.logical_and(mask, pts_pix[0, :] < W - 1)
    mask = np.logical_and(mask, pts_pix[1, :] > 1)
    mask = np.logical_and(mask, pts_pix[1, :] < H - 1)
    points = pts_pix[:, mask]
    depth = pts_cam[2, mask]
    return points, depth

def generate_info_dair(dair_root, split): 
    os.makedirs(os.path.join(dair_root,'depth_gt'), exist_ok=True)
    infos = mmcv.load("data/single-infrastructure-split-data.json")
    split_list = infos[split]
    for sample_id in tqdm(split_list):
        camera_intrinsic_path = os.path.join(dair_root, "calib", "camera_intrinsic", sample_id + ".json")
        virtuallidar_to_camera_path = os.path.join(dair_root, "calib", "virtuallidar_to_camera", sample_id + ".json")
        pcd_path = os.path.join(dair_root, "velodyne", sample_id+".pcd")
        depth_path = os.path.join(dair_root,"depth_gt", sample_id + ".bin")
        r_velo2cam, t_velo2cam = get_velo2cam(virtuallidar_to_camera_path)
        camera_intrinsic = get_P(camera_intrinsic_path)
        points = read_pcd(pcd_path)
        points, depth = obtain_point_depth(points, camera_intrinsic, r_velo2cam, t_velo2cam)
        ply_arr = np.concatenate([points.T, depth[:, None]], axis=1).astype(np.float32)
        ply_arr.flatten().tofile(depth_path)

def main():
    dair_root = "data/dair-v2x-i"
    generate_info_dair(dair_root, split='train')
    generate_info_dair(dair_root, split='val')


if __name__ == '__main__':
    main()
