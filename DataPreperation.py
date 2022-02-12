import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
import cv2
import os

class DataPrepration:
    def __init__(self):
        self.labels_path = 'static/datasets/images-Set1/labels.csv'
        self.images_path = 'static/datasets/images-Set1/'
        pass

    def yolov5_data_prepration(self):
        df = pd.read_csv(self.labels_path)
        print(df.head())
        path = f'{self.images_path}/N2.xml'
        parser = xet.parse(path).getroot()
        name = parser.find('filename').text
        print(name)







dp = DataPrepration()
dp.yolov5_data_prepration()