#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 1: car, 2:trucks, 4: tractors, 5: camping cars, 7: motorcycles, 8:buses, 9: vans, 10: others, 11: pickup, 23: boats , 201: Small Land Vehicles, 31: Large land Vehicles

import os
import pandas as pd
PATH = '/home/data4/zjq/'#'/home/data/zhangjiaqing/dataset/' #chanhe the path firstly (PATH TO dataset)

def update_annotations(filename,image_size,label_path,save_path):
    data = pd.read_csv(label_path + filename, sep=' ', index_col=None, header=None, names=['x_center', 'y_center', 'orientation', 'class', 'is_contained', 'is_occluded', 'corner1_x', 'corner2_x', 'corner3_x', 'corner4_x', 'corner1_y', 'corner2_y', 'corner3_y', 'corner4_y'])

    data['class'].replace(1, 0, inplace=True)
    data['class'].replace(11, 1, inplace=True)
    data['class'].replace(2, 3, inplace=True)
    data['class'].replace(5, 2, inplace=True)
    data['class'].replace(4, 5, inplace=True)
    data['class'].replace(10, 4, inplace=True)
    data['class'].replace(23, 6, inplace=True)
    data['class'].replace(9, 7, inplace=True)
    data['x_center_ratio'] = data['x_center'].astype(float) / image_size
    data['y_center_ratio'] = data['y_center'].astype(float) / image_size
    data['width_ratio'] = (data[['corner1_x', 'corner2_x', 'corner3_x', 'corner4_x']].max(axis=1) - data[['corner1_x', 'corner2_x', 'corner3_x', 'corner4_x']].min(axis=1)) / image_size
    data['height_ratio'] = (data[['corner1_y', 'corner2_y', 'corner3_y', 'corner4_y']].max(axis=1) - data[['corner1_y', 'corner2_y', 'corner3_y', 'corner4_y']].min(axis=1)) / image_size
    res = data.drop(['x_center', 'y_center', 'corner1_x', 'corner2_x', 'corner3_x', 'corner4_x', 'orientation', 'corner1_y', 'corner2_y', 'corner3_y', 'corner4_y', 'is_contained', 'is_occluded'], axis=1)
    res = res.drop(index=res.loc[(res['class'] >7)].index)
    res.to_csv(save_path+ filename, sep=' ', index=False, header=None)

def makelabels():
    label_path = PATH + 'VEDAI/Annotations512'
    save_path = PATH + 'VEDAI/labels'
    list = os.listdir(label_path)
    image_size = 512
    for filename in list:
        update_annotations(filename,image_size,label_path,save_path)


def changepath():
    for i in ['01','02','03','04','05','06','07','08','09','10']:
        path = PATH + 'VEDAI/fold{}.txt'.format(i)
        img_path = PATH + 'VEDAI_1024/images/'
        write_path=(PATH + 'VEDAI/fold{}_write.txt').format(i)
        with open(path, "r") as file:
            img_files = file.readlines()
            for j in range(len(img_files)):
                img_files[j] =  img_path + img_files[j].rstrip()
        file.close()
        with open(write_path, "w") as file:
            for j in range(len(img_files)):
                file.write(img_files[j]+'\n')
        file.close()

        path = PATH + 'VEDAI/fold{}test.txt'.format(i)
        img_path = PATH + 'VEDAI/images/'
        write_path=PATH + 'VEDAI/fold{}test_write.txt'.format(i)
        with open(path, "r") as file:
            img_files = file.readlines()
            for j in range(len(img_files)):
                img_files[j] =  img_path + img_files[j].rstrip()
        file.close()
        with open(write_path, "w") as file:
            for j in range(len(img_files)):
                file.write(img_files[j]+'\n')
        file.close()

if __name__ == '__main__':
    changepath()

