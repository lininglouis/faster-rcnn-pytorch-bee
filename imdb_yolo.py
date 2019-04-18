import numpy as np
import os
import pandas as pd
from roi_data_layer.roidb import combined_roidb
import pickle
import cv2
import glob


imdbval_name = 'colony_2020_test'
imdb, roidb, ratio_list, ratio_index = combined_roidb(imdbval_name, False)
imdb.competition_mode(on=True)

def mkdir_if_not_exists(path):
	if not os.path.exists(path):
		os.makedirs(path)

def mod_all_boxes(all_boxes):

    bg_all_box = []
    for arr in all_boxes[0]:
        bg_all_box.append(arr)

    col_all_box = []
    for arr in all_boxes[1]:
        if arr != []:
            arr = arr[ arr[:, 4] > 0.5]
        col_all_box.append(arr)
    return [ bg_all_box, col_all_box ]
 
def read_test():
    with open('/home/ubuntu/github/frcnn_pytorch_bee/data/colonydevkit2020/colony2020/ImageSets/Main/test.txt', 'r') as f:
        lines = f.readlines()
        lines = [ l.strip() for l in lines ]
        image_dir = '/home/ubuntu/github/frcnn_pytorch_bee/data/colonydevkit2020/colony2020/JPEGImages'
        return { line: os.path.join(image_dir, line +'.JPEG') for line in lines } 

def put_boxes(boxes, img):
    img_canvas = img.copy()
    for box in boxes:
        x1, y1, x2,y2 = box[:4]
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        img_canvas = cv2.rectangle(img_canvas, (x1, y1), (x2, y2), (255,0,0), 2)
    return img_canvas


yolo_pkl_dir = '/home/ubuntu/github/frcnn_pytorch_bee/yolo/yolo_det_pkl/detect*pkl'
yolo_mAP_dir = '/home/ubuntu/github/frcnn_pytorch_bee/yolo/yolo_mAP'

for yolo_pkl in glob.glob(yolo_pkl_dir):
   filename = os.path.basename(yolo_pkl)
   filename = filename.split('_')[1]

   with open(yolo_pkl, 'rb') as f:
       all_boxes = pickle.load(f) 
       all_boxes = mod_all_boxes(all_boxes)
       output_dir = os.path.join(yolo_mAP_dir, 'yolo_mAP_result_{}'.format(filename))
       imdb.evaluate_detections(all_boxes, output_dir) 
       print(yolo_pkl, '---^^^^^^^^^------')
 

paths_dict = read_test()        
from tqdm import tqdm
'''
for idx, (imgname, imgpath) in enumerate(paths_dict.items()):
     boxes = all_boxes[1][idx]
     img = cv2.imread(imgpath)
     labeled_img = put_boxes(boxes, img)
     
     output_dir = 'yolo'
     if not os.path.exists(output_dir):
         os.makedirs(output_dir)
     cv2.imwrite( os.path.join(output_dir, imgname +'.JPEG' ), labeled_img)
'''

