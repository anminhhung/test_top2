import cv2
import numpy as np
import imutils
import argparse
import os 

from shapely.geometry import Point, Polygon

from utils.parser import get_config

def config_cam(img, cfg):
    arr = cfg.CAM.ROI_DEFAULT
    h, w, c = img.shape
    list_lines = []

    # print(line1, line2)
    pts = np.array(arr)

    cv2.drawContours(img, [pts], -1, (0, 0, 255), 2)

    return img

def create_lines_sample_01(cfg):
    arr = cfg.CAM.ROI_DEFAULT
    line2_startX = int((arr[1][0] + arr[2][0]) / 2)
    line2_startY = int((arr[1][1] + arr[2][1]) / 2)
    line2_endX = arr[2][0]
    line2_endY = arr[2][1]
    line2 = ((line2_startX, line2_startY), (line2_endX, line2_endY))

    line1_startX = arr[0][0]
    line1_startY = arr[0][1]
    line1_endX = int((arr[0][0] + arr[3][0]) / 2)
    line1_endY = int((arr[0][1] + arr[3][1]) / 2)
    line1 = ((line1_startX, line1_startY), (line1_endX, line1_endY))

    return [line1, line2]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_cam", type=str, default="./configs/cam.yaml")
    return parser.parse_args()

def create_MOI(img, arr):
    h, w, c = img.shape

    pts = np.array(arr)

    croped = img.copy()

    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

    dst = cv2.bitwise_and(img, croped, mask=mask)
    
    return dst, mask

def save_MOI(image, cam_dir, cam_name, number_MOI):
    subdir = os.path.join(cam_dir, cam_name)
    if not os.path.exists(subdir):
        os.mkdir(subdir)
    
    file_name = os.path.join(subdir, 'MOI_' + str(number_MOI))  
    with open(file_name + '.npy', "wb") as f:
        np.save(f, image)

'''
    Config tay :p
'''
if __name__ == '__main__':
    cam_dir = "cam_MOI"
    if not os.path.exists(cam_dir):
        os.mkdir(cam_dir)

    args = parse_args()
    cfg = get_config()
    cfg.merge_from_file(args.config_cam)
    image = cv2.imread("frame.jpg")

    arr = cfg.CAM1.ROI2
    # dst = config_sample_01(image, cfg)

    dst, mask = create_MOI(image, arr)
    # save_MOI(mask, cam_dir, cfg.CAM1.NAME, 1)

    cv2.imshow("dst.png", dst)
    cv2.waitKey(0)

    # test load image
    # image = np.load('cam_MOI/CAM_01/MOI_1.npy')
    # cv2.imshow("image", image)
    # cv2.waitKey(0)