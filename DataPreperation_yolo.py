import numpy as np
import pandas as pd
from glob import glob
import xml.etree.ElementTree as xet
import cv2
import os
import shutil


class DataPrepration:
    def __init__(self):
        self.df_all = None
        self.label_file = 'static/datasets/images-Set3-yolo/All/labels.csv'
        self.labels_path = 'static/datasets/images-Set3-yolo/All/'
        self.images_path = 'static/datasets/images-Set3-yolo/All/'
        self.train_folder = 'static/datasets/images-Set3-yolo/train'
        self.test_folder = 'static/datasets/images-Set3-yolo/test'

    def parse_label(self, filename):
        path = f'{self.labels_path}{filename}'
        parser = xet.parse(path).getroot()
        name = parser.find('filename').text
        image_size = parser.find('size')
        width = int(image_size.find('width').text)
        height = int(image_size.find('height').text)
        # print(name, width, height)
        return name, width, height

    def findCentre(self):
        df = pd.read_csv(self.label_file)
        df[['img_filename', 'width', 'height']] = df['filepath'].apply(self.parse_label).apply(pd.Series)
        df['center_x'] = (df['xmax'] + df['xmin']) / (2 * df['width'])
        df['center_y'] = (df['ymax'] + df['ymin']) / (2 * df['height'])
        df['bb_width'] = (df['xmax'] - df['xmin']) / df['width']
        df['bb_height'] = (df['ymax'] - df['ymin']) / df['height']
        pd.set_option('display.max_columns', None)
        # print(df.head())
        self.df_all = df
        return df

    def split_train_test(self):
        # split for train and test
        # from 658 images we take 520 for training and the rest for testing
        df_train = self.df_all.iloc[:520]
        df_test = self.df_all.iloc[520:]
        # generate text files for yolo
        values_train = df_train[['img_filename', 'center_x', 'center_y', 'bb_width', 'bb_height']].values
        values_test = df_test[['img_filename', 'center_x', 'center_y', 'bb_width', 'bb_height']].values
        for fname, x, y, w, h in values_train:
            image_name = os.path.split(fname)[-1]
            text_name = os.path.splitext(image_name)[0]
            # copy each image into the training folder
            dst_image_path = os.path.join(dp.train_folder, image_name)
            dst_annotation_file = os.path.join(dp.train_folder, text_name + '.txt')
            shutil.copy(f'{self.images_path}{fname}', dst_image_path)
            label_txt = f'0 {x} {y} {w} {h}'
            print(f'{self.images_path}{fname}')

            with open(dst_annotation_file, mode='w') as file:
                file.write(label_txt)
                file.close()

        for fname, x, y, w, h in values_test:
            image_name = os.path.split(fname)[-1]
            text_name = os.path.splitext(image_name)[0]
            # copy each image into the training folder
            dst_image_path = os.path.join(self.test_folder, image_name)
            dst_annotation_file = os.path.join(self.test_folder, text_name + '.txt')
            shutil.copy(f'{self.images_path}{fname}', dst_image_path)
            label_txt = f'0 {x} {y} {w} {h}'

            with open(dst_annotation_file, mode='w') as file:
                file.write(label_txt)
                file.close()

if __name__ == '__main__':
    dp = DataPrepration()
    df_all = dp.findCentre()
    dp.split_train_test()
