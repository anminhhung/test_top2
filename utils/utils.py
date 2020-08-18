import numpy as np
import cv2
import os 

from shapely.geometry import Point, Polygon
    
# convert compare my_class to GT class
def compare_class(class_id):
    if (class_id >= 0 and class_id <= 4):
        class_id = 0
    if (class_id > 4 and class_id <= 7):
        class_id = 1
    if (class_id == 9 or class_id == 10):
        class_id = 2
    if (class_id == 8 or (class_id <= 13 and class_id > 10)):
        class_id = 3

    return class_id

# GT class name
def get_GT_class_name(class_id):
    class_id = compare_class(class_id)

    if class_id == 0:
        return "LOAI 1"
    elif class_id == 1:
        return "LOAI 2"
    elif class_id == 2:
        return "LOAI 3"
    else:
        return "LOAI 4"
    
def init_board(image, number_MOI=6, col=70, row1=45, row2=65, row3=85, row4=105):
    list_col = []
    cv2.putText(image, "Loai_1:", (5, row1), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
    cv2.putText(image, "Loai_2:", (5, row2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    cv2.putText(image, "Loai_3:", (5, row3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    cv2.putText(image, "Loai_4:", (5, row4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    i = 1
    while i <= number_MOI:
        cv2.putText(image, "MOI_{}".format(i), (col, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        list_col.append(col)

        col += 50
        i += 1

    return image, list_col


def write_board(image, arr_cnt, list_col, number_MOI=6, row1=45, row2=65, row3=85, row4=105):
    list_row = [row1, row2, row3, row4]
    list_color = [(255, 0, 255), (0, 0, 255), (255, 0, 0), (0, 255, 0)]
    col = 1
  
    while col <= number_MOI:
        row = 1
        while row<=4:
            cv2.putText(image, "{}".format(arr_cnt[row-1][col-1]), (list_col[col-1]+20, list_row[row-1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, list_color[row-1], 1)
            row += 1 
        col += 1

    return image

'''
    Check obj in polygon
    input: list vertices
    output: True if in pilygon, otherwise
'''
def check_in_polygon(center_point, polygon):
    pts = Point(center_point[0], center_point[1])
    if polygon.contains(pts):
        return True
    
    return False

'''
    return mask(ROI) of MOI
'''
def check_number_MOI(number, cfg):
    number_MOI = cfg.CAM.NUMBER_MOI
    if number_MOI == 2:
        switcher = {
            1: Polygon(cfg.CAM.ROI1),
            2: Polygon(cfg.CAM.ROI2)
        }
    if number_MOI == 6:
        switcher = {
            1: Polygon(cfg.CAM.ROI1),
            2: Polygon(cfg.CAM.ROI2),
            3: Polygon(cfg.CAM.ROI3),
            4: Polygon(cfg.CAM.ROI4), 
            5: Polygon(cfg.CAM.ROI5),
            6: Polygon(cfg.CAM.ROI6)
        }

    return switcher.get(number, "Invalid ROI of cam")
