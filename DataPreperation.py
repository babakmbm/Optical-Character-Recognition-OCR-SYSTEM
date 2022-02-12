import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
import cv2
import os

class DataPrepration:
    def __init__(self):
        self.labels_path = 'static/datasets/images-Set1/labels.csv'
        pass

    def yolov5_data_prepration(self):
        df = pd.read_csv(self.labels_path)
        print(df.head())







dp = DataPrepration()
dp.yolov5_data_prepration()