gt_dir = '/workspace/mnt/storage/guangcongzheng/zju_wwj_backup/dair-v2x-i-kitti/demo_gt'
bevheight_dir = '/workspace/mnt/storage/guangcongzheng/zju_wwj_backup/dair-v2x-i-kitti/demo_height'
bevspread_dir = '/workspace/mnt/storage/guangcongzheng/zju_wwj_backup/dair-v2x-i-kitti/demo_spread'
compare_dir = '/workspace/mnt/storage/guangcongzheng/zju_wwj_backup/dair-v2x-i-kitti/compare'
import os
import cv2
from tqdm import tqdm
if __name__ == "__main__":
    for image_file in tqdm(os.listdir(bevheight_dir)):
        image_gt_path = os.path.join(gt_dir,image_file)
        image_height_path = os.path.join(bevheight_dir,image_file)
        image_spread_path = os.path.join(bevspread_dir,image_file)
        image_gt = cv2.imread(image_gt_path)
        image_height = cv2.imread(image_height_path)
        image_spread = cv2.imread(image_spread_path)
        compare = cv2.hconcat([image_gt,image_height,image_spread])
        compare_path = os.path.join(compare_dir,image_file)
        cv2.imwrite(compare_path,compare)
