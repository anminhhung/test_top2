from __future__ import division, print_function, absolute_import

from timeit import time
from PIL import Image
import warnings
import cv2
import numpy as np
import argparse
import os
import shutil
import imutils.video

from shapely.geometry import Point, Polygon
from collections import deque

from src.detect import build_detector_v3
from videocaptureasync import VideoCaptureAsync
from utils.parser import get_config
from utils.utils import compare_class, get_GT_class_name

from Tracktor import Track, get_center, get_height, get_width, make_pos

class VideoTracker(object):

    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.video_path = args.VIDEO_PATH
        self.detector = build_detector_v3(cfg)
        self.motion_model_cfg = cfg.MOTION_MODEL

        # set the values to
        # self.reset()
        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.results = {}
        self.countor_restults = {}
        self.im_index = 0
    
    # def reset(self, hard=True):
    #     self.tracks = []
    #     self.inactive_tracks = []

    #     if hard:
    #         self.track_num = 0
    #         self.results = {}
    #         self.countor_restults = {}
    #         self.im_index = 0

    def tracks_to_inactive(self, tracks):
        self.tracks = [track for track in self.tracks if track not in tracks]

        for track in tracks:
            track.position = track.last_position[-1]
            # tobe continue


    '''
        init new tracks and save them
    '''
    def add(self, new_det_positions, new_det_scores): # num_det_positions and scores are numpy array
        print("-----------")
        print("[INFO] new det positions: ", new_det_positions)
        num_new_track = np.size(new_det_positions, axis=0)
        # print("[INFO] check add")
        for i in range(num_new_track):
            self.tracks.append(Track(
                new_det_positions[i],
                new_det_scores[i],
                self.track_num,
                self.motion_model_cfg.n_steps
            ))
            self.track_num += 1
            # print("[INFO] tracks[i]: ", self.tracks[i].position)
            # self.tracks[i].add_last_position(new_det_positions[i])

        for track in self.tracks:
            print("[INFO] tracks[i].position: ", track.position)
        # self.track_num += num_new_track
    
    ''' 
        get the positions of all active tracks
    '''
    def get_pos(self):
        if len(self.tracks) == 1:
            positions =  self.tracks[0].position
            ids = self.tracks[0].id
        elif len(self.tracks) > 1:
            positions = np.array([_track.position for _track in self.tracks])
            ids = np.array([_track.id for _track in self.tracks])
            print("******************")
            print("position in self.get_pos: ", positions)
            print("ids in self.get pos: ", ids)
            print("******************")
        else:
            positions = np.empty(0)
            ids = np.empty(0)
        
        return positions, ids
    
    '''
        updates the given track's position by one step based on track.last_v
    '''
    def motion_step(self, track):
        if self.motion_model_cfg.center_only:
            center_new = get_center(track.position) +  track.last_v
            track.position = make_pos(*center_new, get_width(track.position), get_height(track.position))
        else:
            track.position += track.last_v

    '''
        Applies a simple linear motion model that considers the last n_steps steps
    '''
    def motion(self):
        print("[INFO] check in motion has len tracks: ", len(self.tracks))
        for track in self.tracks:
            last_position = list(track.last_position)
            # avg velocity between each pair of consecutive positions in track.last_position
            if len(last_position) == 2:
                if self.motion_model_cfg.center_only:
                    vs = get_center(last_position[0])  - get_center(last_position[1])
                else:
                    vs = last_position[0] - last_position[1]
            else:
                vs = np.array([0, 0, 0, 0], dtype=int)
        
            track.last_v = np.array(vs).mean()
        
            self.motion_step(track)

    '''
        Remove boxes which area is smaller than min_size
        input: 
            - boxes: boxes in (x1, y1, x2, y2) format
            - min_size (float): minimum size
        output:
            keep: indices of the boxes that have both sides larger than min_size
    '''
    def remove_boxes_out_roi(self, boxes, ROI_image, min_in_porcentage):
        area_in = [ROI_image[box[1]:box[3], box[0]:box[2]].sum() for box in boxes]
        area_in = np.stack(area_in)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        procentage_in = area_in / area.float()
        keep = procentage_in >= min_in_porcentage
        keep = keep.nonzero()

        return keep

    '''
        Remomve boxes which area is smaller than min_size
        input:
            - boxes: boxes in (x1, y1, x2, y2) format
            - min_size: minimum_size
        output:
            - keep: indices of the boxes that have both sides larger than min_size
    '''
    def remove_small_boxes_area(self, boxes, min_size):
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # keep = area >= min_size
        keep = [i for i in range(len(area)) if area[i] > min_size]
        # keep = keep.nonzero()

        return keep
    
    '''
        Remove boxes which area is biger than max_size
        input:
            - boxes: boxes in (x1, y1, x2, y2) format
            - max_size: maximum_size
        output:
            - keep: indices of the boxes that have smaller than max_size
    '''
    def remove_big_boxes_area(self, boxes, max_size):
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        # keep = area <= max_size
        keep = [i for i in range(len(area)) if area[i] > max_size]
        # keep = keep.nonzero()

        return keep
    
    '''
        Remove low scoring boxes
        input:
            - arr_scores: arr scores for each bbox
            - min_score: minimum_score
        output:
            - keep: indices of the boxes that have both sides larger than min_score
    '''
    def remove_low_score(self, arr_scores, min_score):
        # keep = arr_scores >= min_score
        keep = [i for i in range(len(arr_scores)) if arr_scores[i] > min_score]
        # keep = keep.nonzero()

        return keep

    ''' 
        None max supression
        input:
            - boxes: list bbox, scores, labels
            - overlapThreshold: thresh of overlapping
    '''
    def nms(self, boxes, scores, overlapThreshold):
        # if none bboxes => return empty list
        # print("[INFO] bboxes: ", boxes)
        # print("[INFO] scores: ", scores)
        if len(boxes) == 0:
            return []
        
        # if bboxes int -> convert int to float
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")
        
        keep = []

        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
       
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        while len(idxs) > 0:
            last_index = len(idxs) - 1
            i = idxs[last_index]
            keep.append(i)

            # find the max: A(x, y) and min: B(x, y). A, B: max, min points of bbox
            max_x1 = np.maximum(x1[i], x1[idxs[:last_index]])
            max_y1 = np.maximum(y1[i], y1[idxs[:last_index]])
            min_x2 = np.minimum(x2[i], y2[idxs[:last_index]])
            min_y2 = np.minimum(y2[i], y2[idxs[:last_index]])

            # compute width height of bbox
            w = np.maximum(0, min_x2 - max_x1 + 1)
            h = np.maximum(0, min_y2 - max_y1 + 1)
            # box_width = width[i]
            # box_heigth = height[i]

            # print("[INFO] area[idxs[:last_index]]: ", area[idxs[:last_index]])
            # compute ratio of S overlap
            S_overlap = (w * h) / area[idxs[:last_index ]]
            # remove last index and indexes has S_overlap > threshold
            idxs = np.delete(idxs, np.concatenate(([last_index],
                    np.where(S_overlap > overlapThreshold)[0])))
                    
        boxes = boxes[keep]
        scores = scores[keep]

        # print("[INFO] after nsm: ", boxes)
        # print("[INFO] scores after nsm: ", scores)

        # remove elements has score=2
        keep  = scores != 2.

        # print("[INFO] boxes after remove 2: ", boxes[keep])
        # print("[INFO] scores after remove 2: ", scores[keep])

        return boxes[keep].astype("int"), scores[keep]

    def check_intersect_reg(self, reg1, reg2):
        x1_start, y1_start, x1_end, y1_end = reg1[:]
        # print("[INFO] reg2: ", reg2)
        x2_start, y2_start, x2_end, y2_end = reg2[:]

        w1 = x1_end - x1_start
        w2 = x2_end - x2_start

        if (x1_start + w1 >= x2_start) and (x2_start + w2 >= x1_start) and (y1_start + w1 >= y2_start) and (y2_start + w2 >= y1_start):
            print("[INFO] check intersect reg True")
            return True
        
        return False

    def bb_intersection_over_union(self, boxA, boxB):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def check_nms(self, track_box, track_score, index_track, det_boxes, det_scores):
        # print("[INFO] track_Box: ", track_box)
        len_det_boxes = len(det_boxes)
        for i in range(len_det_boxes):
            if self.check_intersect_reg(track_box, det_boxes[i]):
                IOU = self.bb_intersection_over_union(track_box, det_boxes[i])
                print("[INFO] IOU: ", IOU)
                if IOU > self.motion_model_cfg.nms_thresh:
                    # print("[INFO] det_boxes[i]: ", det_boxes[i])
                    if track_score < det_scores[i]:
                        self.tracks[index_track].position = det_boxes[i] 
                        self.tracks[index_track].score = det_scores[i]
                
                    det_boxes = np.delete(det_boxes, i, axis=0)
                    det_scores = np.delete(det_scores, i)
                    # print("[INFO] remove: ", i)
                    # print("-------------")
                    # print("[INFO] det_boxes after remove: \n", det_boxes)

                    return det_boxes, det_scores
                
        # print("-------------")
        # return det_boxes, det_scores

    def resize_box(self, box, original_size, new_size):
        ratios = (new_size[0]/original_size[0], new_size[1]/original_size[1])

        ratio_height, ratio_width = ratios
        x_min, y_min, x_max, y_max = box

        x_min *= ratio_width
        x_max *= ratio_width
        y_min *= ratio_height
        y_max *= ratio_height

        return np.array([x_min, y_min, x_max, y_max])

    # Tracktor
    def tracking_processing(self, track_boxes, boxes_ids, image, _image, ROI_image):
        # if len(track_boxes)>0:
            boxes, scores, labels = self.detector(image)
            # print("len_ids: ", len(boxes_ids))
            boxes = np.array(boxes)
            scores = np.array(scores)
            labels = np.array(labels) 

            print("[INFO] len boxes before nms: ", len(boxes))
            boxes, scores = self.nms(boxes, scores, self.motion_model_cfg.nms_thresh)
            print("[INFO] len boxes after nms: ", len(boxes))

            # for box in boxes:
            #     print("[INFO] RUN DETECTION CHECK SHAPE box: ", box.shape)

            # print("[INFO] boxes in tracking_processing: ", boxes)

            # print("[INFO] len bboxs after remove low score: ", len(boxes))
            # # remove low score boxes
            # keep = self.remove_low_score(scores, self.motion_model_cfg.detection_object_thresh)
            # boxes, scores, labels, boxes_ids = boxes[keep], scores[keep], labels[keep], boxes_ids[keep]

            # # remove small boxes
            # print("[INFO] len bboxs after remove small boxes: ", len(boxes))
            # keep = self.remove_small_boxes_area(boxes, self.motion_model_cfg.detection_min_area_val)
            # print("[INFO] keep: ", keep)
            # boxes, scores, labels, boxes_ids = boxes[keep], scores[keep], labels[keep], boxes_ids[keep]

            # #remove big boxes
            # keep = self.remove_big_boxes_area(boxes, self.motion_model_cfg.detection_max_area_val)
            # boxes, scores, labels, boxes_ids = boxes[keep], scores[keep], labels[keep], boxes_ids[keep]

            # remove boxes out ROI 
            # if np.size(boxes):
            #     keep = self.remove_boxes_out_roi(boxes, ROI_image, self.motion_model_cfg.detection_ROI_in)
            #     boxes, scores, labels, boxes_ids = boxes[keep], scores[keep], labels[keep], boxes_ids[keep]
            
            # indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)

            return boxes, scores, labels, boxes_ids
        # else:
        #     return np.empty(0), np.empty(0), np.empty(0), np.empty(0)

    def run_detection(self, image, _image, ROI_image, track_boxes, track_scores):
        boxes, scores, labels = self.detector(image)
        
        boxes = np.array(boxes)
        scores = np.array(scores)
        labels = np.array(labels) 

        # for box in boxes:
        #     print("[INFO] RUN DETECTION CHECK SHAPE box: ", box.shape)    

        # for boxes, scores, labels, ROI_image, track_boxes_b, image, original_img_shape in zip(
        #                                                                 arr_boxes, arr_confidences,
        #                                                                 arr_classes, ROI_images, track_boxes,
        #                                                                 images, original_image_sizes
        #                                                             ):

        # remove low score boxes
        # keep = self.remove_low_score(scores, self.motion_model_cfg.detection_object_thresh)
        # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # # remove small boxes
        # keep = self.remove_small_boxes_area(boxes, self.motion_model_cfg.detection_min_area_val)
        # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # remove big boxes
        # keep = self.remove_big_boxes_area(boxes, self.motion_model_cfg.detection_max_area_val)
        # boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        # remove boxes out ROI 
        # if np.size(boxes):
        #     keep = self.remove_boxes_out_roi(boxes, ROI_image, self.motion_model_cfg.detection_ROI_in)
        #     boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
            
        # filter detection in track
        # for i in range(len(boxes)):
        #     print("boxes: ", boxes[i])

        # if len(track_boxes) > 0:
        #     print("[INFO] len track boxes: ", len(track_boxes))
        #     for track_box in track_boxes:
        #         track_box = track_box.reshape((1, 4))
        #         # print("[INFO] boxes before: ", boxes)
        #         # print("[INFO] track box: ", track_box)
        #         # print("[INFO] labels before: ", labels)
        #         boxes_det_track = np.r_[boxes, track_box]
        #         print("[INFO] boxes det_track: ", boxes_det_track)
        #         # len_track_boxes = np.size(track_box, axis=0)
        #         # pick 2 b/c range of score: 0->1
        #         scores_det_track = np.r_[scores, np.ones(1) * 2]
        #         # applies nms
        #         # print("[INFO] len track before nms: ", boxes)
        #         # print("[INFO] boxes_det_track: ", boxes_det_track)
        #         boxes, scores = self.check_nms(boxes_det_track, scores_det_track, self.motion_model_cfg.nms_thresh)
                # boxes, scores = boxes[keep_boxes], scores[keep_scores]
        for i in range(len(track_boxes)):
            boxes, scores = self.check_nms(track_boxes[i], track_scores[i], i, boxes, scores)

        # else:
        #     print("[INFO]len(track_boxes) = 0 add boxes to list track")
        #     self.add(boxes, scores)

        print("[INFO] boxes after: ", len(boxes))
        # print("[INFO] labels after: ", labels)
        # indexes = cv2.dnn.NMSBoxes(boxes, scores, 0.5, 0.4)
        # for i in range(len(boxes)):
        #     # if i in indexes:
        #     x1, y1, x2, y2 = boxes[i]
        #     class_name = str(labels[i])
        #     cv2.rectangle(_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
        #     cv2.putText(_image, class_name + ": " + str(round(scores[i], 2)), (x1, y1), \
        #                 cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 1)
    
        return image, boxes, scores, labels
    
    def run_track_and_detect(self, image, _image, boxes, boxes_ids, ROI_image, count_frame):
        # get track
        track_boxes, track_scores, track_labels, track_ids = self.tracking_processing(boxes, boxes_ids, image, _image, ROI_image)
        
        # add track if first frame
        if count_frame == 1:
            self.add(track_boxes, track_scores)
            track_ids = np.arange(len(track_boxes)) 

        # get detect
        image, det_boxes, det_scores, det_labels = self.run_detection(image, _image, ROI_image, track_boxes, track_scores)
        
        print("################")
        print("len track_boxes: ", len(track_boxes))
        print("len track_scores: ", len(track_scores))
        print("len track ids: ", len(track_ids))
        print("###############")
        # draw
        for i in range(len(track_boxes)):
            try:
                x1, y1, x2, y2 = track_boxes[i]
                class_name = str(track_ids[i])
                cv2.rectangle(_image, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(_image, class_name + ": " + str(round(track_scores[i], 2)), (x1, y1), \
                            cv2.FONT_HERSHEY_SIMPLEX,1, (255, 255, 0), 1)
            except:
                pass

        return _image, track_boxes, track_scores, track_labels, track_ids, det_boxes, det_scores, det_labels

    def process(self, image, _image, count_frame):
        for track in self.tracks:
            # add current position to last_position 
            track.last_position.append(track.position)

        # for track in self.tracks:
        #     print("track id: ", track.id)
        # apply motion model
        if len(self.tracks):
            if self.motion_model_cfg.enabled:
                self.motion()
                self.tracks = [track for track in self.tracks if track.has_positive_area()]
        
        positions, ids = self.get_pos()
        # print("[INFO] len(position): {}, len(ids): {} after get_pos".format(len(positions), len(ids)))

        # run detect and track
        image, track_boxes, track_scores, track_labels, track_ids, det_boxes, det_scores, det_labels = self.run_track_and_detect(image, _image, positions, ids, self.motion_model_cfg.ROI_DEFAULT, count_frame)
        print("############")
        print("[INFO] track ID: ", track_ids)
        print("############")
        # for track in track_boxes:
        #     print("[INFO] track shape: ", track.shape)
        # update position and score for the current track
        # track_to_inactive_list = []
        # if len(self.tracks):
        #     print("[INFO] update track and score")
        #     print("[INFO] len track boxes in update box and score: ", len(self.tracks))
        #     print("[INFO] track boxes: ", track_boxes)
        #     print("[INFO] track scores: ", track_scores)
        #     for i in range(len(self.tracks)):
        #         if self.tracks[i].id in track_ids:
        #             # print("[INFO] self.tracks[i].id: ", self.tracks[i].id)
        #             # index = track_ids == self.tracks[i].id 
        #             # index = index.nonzero()
        #             # index = i for i in range(len(track_ids)) if track_ids[i] == self.tracks[i].id
        #             for j in range(len(track_ids)):
        #                 if track_ids[j] == self.tracks[i].id:
        #                     # index = i
        #                     print("[INFO] track box update : ", track_boxes[j])
        #                     self.tracks[i].position = track_boxes[j]
        #                     print("[INFO] tracks[i].position update: ", self.tracks[i].position)
        #                     self.tracks[i].score = track_scores[j]
        #                     break
        
        #             # self.tracks[i].position = track_boxes[index]
        #             # self.tracks[i].score = track_scores[index]
   
        #         else:
        #             # if the index is not in the ids set to inactive
        #             track_to_inactive_list.append(self.tracks[i])
        
        # set the intective track (còn 1 dòng)

        # create new track
        print("[INFO] len det boxes in create new track: ", len(det_boxes))
        if len(det_boxes) > 0:
            self.add(det_boxes, det_scores)

        print("[INFO] len self.tracks: ", len(self.tracks))

        return image

    def run(self):
        asyncVideo_flag = self.args.asyncVideo_flag
        writeVideo_flag = self.args.writeVideo_flag

        if asyncVideo_flag:
            video_capture = VideoCaptureAsync(self.video_path)
        else:
            video_capture = cv2.VideoCapture(self.video_path)

        if asyncVideo_flag:
            video_capture.start()

        if writeVideo_flag:
            if asyncVideo_flag:
                w = int(video_capture.cap.get(3))
                h = int(video_capture.cap.get(4))
            else:
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter('output_yolov4.avi', fourcc, 30, (w, h))
            frame_index = -1
        
        fps = 0.0
        fps_imutils = imutils.video.FPS().start()

        count_frame = 0
        while True:
            count_frame += 1
            t1 = time.time()
            ret, frame = video_capture.read()  
            if ret != True:
                break
            
            _frame = frame
            # process 
            frame = self.process(frame, _frame, count_frame)

            # visualize
            if self.args.visualize:
                frame = imutils.resize(frame, width=1000)
                cv2.imshow("Final result", frame)
            
            if writeVideo_flag:  # and not asyncVideo_flag:
                # save a frame
                out.write(frame)
                frame_index = frame_index + 1
            
            fps_imutils.update()

            if not asyncVideo_flag:
                fps = (fps + (1./(time.time()-t1))) / 2
                print("FPS = %f" % (fps))

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        fps_imutils.stop()
        print('imutils FPS: {}'.format(fps_imutils.fps()))

        if asyncVideo_flag:
            video_capture.stop()
        else:
            video_capture.release()

        if writeVideo_flag:
            out.release()

        cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("VIDEO_PATH", type=str)
    parser.add_argument("--config_detection", type=str, default="./configs/yolov3.yaml")
    parser.add_argument("--motion", type=str, default="./configs/motion.yaml")
    # parser.add_argument("--config_cam", type=str, default="./configs/cam1.yaml")
    parser.add_argument("-v", "--visualize", type=bool, default=False)
    parser.add_argument("-w", "--writeVideo_flag", type=bool, default=False)
    parser.add_argument("--asyncVideo_flag", type=bool, default=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = get_config()
    # setup code
    cfg.merge_from_file(args.config_detection)
    cfg.merge_from_file(args.motion)

    # run code 
    video_tracker = VideoTracker(cfg, args)
    video_tracker.run()