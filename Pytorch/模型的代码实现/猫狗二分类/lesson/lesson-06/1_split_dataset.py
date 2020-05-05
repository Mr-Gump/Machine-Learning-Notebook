# -*- coding: utf-8 -*-
"""
# @file name  : 1_split_dataset.py
# @author     : tingsongyu
# @date       : 2019-09-07 10:08:00
# @brief      : 将数据集划分为训练集，验证集，测试集
"""

import os
import random
import shutil

# 创建文件夹
def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

# 将图片分类
def classify(pre_dir,dataset_dir,cat_dir,dog_dir):
    for root, dirs, files in os.walk(pre_dir):
        for name in files:
            if name[:3] == 'cat':
                shutil.copy(os.path.join(root,name),cat_dir)
            else :
                shutil.copy(os.path.join(root,name),dog_dir)


if __name__ == '__main__':

    random.seed(1)    # 生成种子

    pre_dir = os.path.join("..", "..", "data","DC_pre_data")
    dataset_dir = os.path.join("..", "..", "data", "DC_data")     # 数据路径
    cat_dir = os.path.join(dataset_dir,"cat")   # 猫图片路径
    dog_dir = os.path.join(dataset_dir,"dog")   # 狗图片路径
    makedir(dataset_dir)     # 创建数据文件夹
    makedir(cat_dir)         # 创建猫图片文件夹
    makedir(dog_dir)         # 创建狗图片文件夹
    classify(pre_dir,dataset_dir,cat_dir,dog_dir)
    split_dir = os.path.join("..", "..", "data", "dc_split")    # 分割后的路径
    train_dir = os.path.join(split_dir, "train")    # 训练集
    valid_dir = os.path.join(split_dir, "valid")    # 验证集
    test_dir = os.path.join(split_dir, "test")      # 测试集

    train_pct = 0.8     # 训练集百分比
    valid_pct = 0.1     # 验证集百分比
    test_pct = 0.1      # 测试集百分比

    for root, dirs, files in os.walk(dataset_dir):    # 遍历数据文件夹,root:根文件夹 dirs:子文件夹 files:文件

        for sub_dir in dirs:    # 子文件夹

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))  # 提取图片名

            random.shuffle(imgs)   # 乱序排列
            img_count = len(imgs)  # 获取图片数量

            train_point = int(img_count * train_pct)    # 训练集的数量
            valid_point = int(img_count * (train_pct + valid_pct))   # 验证集的数量

            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)    # 训练集输出文件夹
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)    # 验证集输出文件夹
                else:
                    out_dir = os.path.join(test_dir, sub_dir)      # 测试集输出文件夹

                makedir(out_dir)     # 创建文件夹

                target_path = os.path.join(out_dir, imgs[i])     # 图片目标路径
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])   # 图片原路径
                shutil.copy(src_path, target_path)    # 复制图片

            print('Class:{}, train:{}, valid:{}, test:{}'.format(sub_dir, train_point, valid_point-train_point,
                                                                 img_count-valid_point))
