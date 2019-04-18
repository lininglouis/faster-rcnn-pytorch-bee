import os
import pickle

det_file = './output/res101/colony_2020_test/faster_rcnn_10/detections.pkl'


with open(det_file, 'rb') as f:
    data = pickle.load(f)
